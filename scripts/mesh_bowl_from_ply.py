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
    p.add_argument("--keep_largest_only", action="store_true", default=True,
                   help="After meshing, keep only the largest connected component (drops artifact fragments).")
    p.add_argument("--fill_holes", action="store_true", default=False,
                   help="Fill holes in the mesh after cleanup. (Disabled by default: open3d's fill_holes can emit numerically-corrupt vertices on noisy Poisson output.)")
    p.add_argument("--bake_vertex_colors", action="store_true", default=True,
                   help="Transfer the .ply's per-Gaussian color (SH DC f_dc_0/1/2) to mesh vertices via nearest neighbor.")
    p.add_argument("--precompute_coacd", action="store_true", default=True,
                   help="Run CoACD now and write the cache file (<out_obj>.coacd.ply) alongside so env-build time stays fast.")
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

    # Drop everything except the largest connected component (removes floating
    # artifact fragments from Poisson around the bowl rim).
    if args.keep_largest_only:
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n = np.asarray(cluster_n_triangles)
        if len(cluster_n) > 0:
            largest = int(cluster_n.argmax())
            keep = triangle_clusters == largest
            mesh.remove_triangles_by_mask(~keep)
            mesh.remove_unreferenced_vertices()
            print(f"after largest-CC keep: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris "
                  f"({len(cluster_n)} components -> 1)")

    # Decimate
    if args.target_n_faces and len(mesh.triangles) > args.target_n_faces:
        mesh = mesh.simplify_quadric_decimation(args.target_n_faces)
        print(f"after decimate: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Fill holes (open3d 0.19 has TriangleMesh.fill_holes from the t.geometry API).
    if args.fill_holes:
        try:
            t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            t_mesh = t_mesh.fill_holes(hole_size=0.05)
            mesh = t_mesh.to_legacy()
            print(f"after fill_holes: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")
        except Exception as e:
            print(f"fill_holes skipped: {e}")

    mesh.compute_vertex_normals()

    # Bake per-vertex colors from the .ply's SH DC coefficients (f_dc_0/1/2).
    # The Gaussian Splatting convention is: rgb = sigmoid(0.5 + 0.282 * f_dc)
    # but most viewers just use rgb = clip(0.5 + 0.282 * f_dc, 0, 1).
    if args.bake_vertex_colors:
        try:
            from plyfile import PlyData
            from scipy.spatial import cKDTree
            ply = PlyData.read(args.in_ply)
            v = ply["vertex"]
            pts_ply = np.stack([v["x"], v["y"], v["z"]], axis=1)
            fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
            rgb = np.clip(0.5 + 0.282 * fdc, 0.0, 1.0)
            tree = cKDTree(pts_ply)
            mesh_verts = np.asarray(mesh.vertices)
            _, nn_idx = tree.query(mesh_verts, k=1)
            mesh.vertex_colors = o3d.utility.Vector3dVector(rgb[nn_idx])
            print(f"baked vertex colors from .ply SH DC, mean RGB = {rgb[nn_idx].mean(0)}")
        except Exception as e:
            print(f"vertex color bake skipped: {e}")

    pts_after = np.asarray(mesh.vertices)
    extent = pts_after.max(0) - pts_after.min(0)
    print(f"final extent (m): {extent}")

    # Re-origin the mesh: xy centered at 0, z min at 0 (bowl floor).
    # This makes the mesh drop in at table height with no extra offset.
    pts_after = np.asarray(mesh.vertices)
    xy_center = 0.5 * (pts_after[:, :2].max(0) + pts_after[:, :2].min(0))
    z_min = pts_after[:, 2].min()
    shift = np.array([-xy_center[0], -xy_center[1], -z_min])
    mesh.translate(shift, relative=True)
    print(f"shifted by {shift.tolist()} so origin is at bowl bottom-center")

    out_dir = os.path.dirname(args.out_obj)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # .obj for collision (CoACD-friendly), .ply alongside for visuals
    # because .obj's vertex-color extension isn't portable.
    o3d.io.write_triangle_mesh(args.out_obj, mesh, write_ascii=False)
    ply_out = os.path.splitext(args.out_obj)[0] + ".ply"
    o3d.io.write_triangle_mesh(ply_out, mesh, write_ascii=False,
                               write_vertex_colors=True, write_vertex_normals=True)
    print(f"wrote {args.out_obj} and {ply_out}")

    # Precompute CoACD so the env-build path hits a cached decomposition
    # instead of running CoACD on first env construction (~10-30 s otherwise).
    if args.precompute_coacd:
        try:
            from sapien.wrapper.coacd import do_coacd
            out_path = do_coacd(args.out_obj)
            print(f"coacd cache ready at {out_path}")
        except Exception as e:
            print(f"coacd precompute skipped: {e}")


if __name__ == "__main__":
    main()
