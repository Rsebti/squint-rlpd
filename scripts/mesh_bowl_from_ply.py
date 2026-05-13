"""Mesh the SAM-3D Gaussian-splat bowl .ply into a watertight .obj.

Reads the 337k Gaussian centers + their normals, runs Poisson surface
reconstruction, trims low-density artifacts, and writes the result as a
triangle mesh that SAPIEN/PhysX can load for collision (after CoACD
decomposition) and visual.

Usage:
    python scripts/mesh_bowl_from_ply.py \
        --in_ply=sam3d/bowl_raw_output_scaled.ply \
        --out_obj=envs/meshes/bowl.obj
"""
import argparse
import os

import numpy as np
import open3d as o3d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_ply", required=True)
    p.add_argument("--out_obj", required=True)
    p.add_argument("--depth", type=int, default=9,
                   help="Poisson reconstruction octree depth. 8-10 typical.")
    p.add_argument("--density_quantile", type=float, default=0.02,
                   help="Drop the bottom q-quantile of vertex densities to remove balloon artifacts.")
    p.add_argument("--target_n_faces", type=int, default=8000,
                   help="Decimate the final mesh to this many faces for sim performance.")
    args = p.parse_args()

    pcd = o3d.io.read_point_cloud(args.in_ply)
    pts = np.asarray(pcd.points)
    print(f"loaded {len(pts)} points  bbox extent = {(pts.max(0)-pts.min(0))}")
    # SAM 3D writes the nx/ny/nz fields as 0 — re-estimate normals from the
    # local geometry.
    n_in = np.asarray(pcd.normals)
    if len(n_in) == 0 or float(np.linalg.norm(n_in, axis=1).mean()) < 0.1:
        print("estimating normals (PLY had zero normals)")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )
        # Orient so the bowl's outside points outward (consistent orientation
        # helps Poisson)
        pcd.orient_normals_consistent_tangent_plane(k=30)

    # Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=args.depth
    )
    print(f"poisson: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    # Density-based pruning (removes the floating-balloon vertices that
    # Poisson invents outside the actual support)
    densities = np.asarray(densities)
    keep_mask = densities >= np.quantile(densities, args.density_quantile)
    mesh.remove_vertices_by_mask(~keep_mask)
    print(f"after density prune: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    # Crop to the original point cloud's AABB (kills artifacts beyond the bowl)
    aabb = pcd.get_axis_aligned_bounding_box()
    margin = 0.005
    aabb.min_bound = aabb.min_bound - margin
    aabb.max_bound = aabb.max_bound + margin
    mesh = mesh.crop(aabb)
    print(f"after bbox crop: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    # Decimate
    if args.target_n_faces and len(mesh.triangles) > args.target_n_faces:
        mesh = mesh.simplify_quadric_decimation(args.target_n_faces)
        print(f"after decimate: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    mesh.compute_vertex_normals()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    pts_after = np.asarray(mesh.vertices)
    extent = pts_after.max(0) - pts_after.min(0)
    print(f"final extent (m): {extent}")

    out_dir = os.path.dirname(args.out_obj)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    o3d.io.write_triangle_mesh(args.out_obj, mesh, write_ascii=False)
    print(f"wrote {args.out_obj}")


if __name__ == "__main__":
    main()
