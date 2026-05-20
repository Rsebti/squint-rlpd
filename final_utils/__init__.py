"""Reusable pick-and-place for the SO101, for use across evals.

The picking policy + FK-gated hardcoded grasp (solved 2026-05-20) plus an IK
place phase, wrapped in one callable:

    from final_utils import pick_and_place
    ok = pick_and_place(goal_color=0, bowl_xy=(0.25, 0.10))

See final_utils/pick_place.py.
"""
from .pick_place import pick_and_place

__all__ = ["pick_and_place"]
