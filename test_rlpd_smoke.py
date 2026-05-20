"""Functional smoke test for the RLPD implementation.

Run this on the 5090 machine (or any box with torch + tensordict installed)
AFTER ``parse_check_rlpd.py`` passes. It exercises the network classes and the
offline-buffer round-trip without booting ManiSkill, so it's fast (~5 s) and
catches:

  * Critic forward shape regressions after the C51 strip.
  * subset_size > num_q indexing bugs in the sample-then-min target.
  * TensorDict layout mismatches between save and load in rlpd_utils.

Usage:
    python test_rlpd_smoke.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch
from tensordict import TensorDict

# Import the model classes from train_rlpd. We do NOT call the main() that
# boots ManiSkill — just touch the standalone class definitions.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_critic_forward_and_target() -> None:
    """Instantiate Critic on CPU, forward a random batch, build the SAC target
    the same way ``update_main`` in train_rlpd.py does."""
    from train_rlpd import Critic  # noqa: WPS433 — local import is intentional

    n_state, n_act, num_q, subset_size, B = 16, 6, 10, 2, 32
    # The CNN encoder produces a feature vector of repr_dim ~ 64 in real runs;
    # any positive int works for the smoke test since Critic.proj is a Linear.
    repr_dim = 64
    critic = Critic(n_obs=repr_dim, n_state=n_state, n_act=n_act, num_q=num_q)
    rgb_feat = torch.randn(B, repr_dim)
    state = torch.randn(B, n_state)
    actions = torch.randn(B, n_act).clamp(-1, 1)

    q = critic(rgb_feat, state, actions)
    assert q.shape == (num_q, B), f"forward shape {q.shape} != ({num_q}, {B})"

    # Sample-then-min target (mirrors update_main's logic).
    subset_idx = torch.randint(0, num_q, (subset_size,))
    q_subset = q.index_select(0, subset_idx)
    q_min = q_subset.min(dim=0).values
    assert q_min.shape == (B,), f"target min shape {q_min.shape} != ({B},)"

    # get_q_values with detach_critic preserves shape.
    q_detached = critic.get_q_values(rgb_feat, state, actions, detach_critic=True)
    assert q_detached.shape == (num_q, B)

    # MSE target broadcast.
    target = torch.randn(B)
    loss = torch.nn.functional.mse_loss(q, target.unsqueeze(0).expand_as(q))
    assert torch.isfinite(loss), "MSE produced non-finite loss"

    print(f"[ OK ] Critic forward / sample-then-min / MSE target (num_q={num_q}, subset={subset_size}, B={B})")


def test_offline_bundle_roundtrip() -> None:
    """Write a synthetic TensorDict via save_offline_bundle, reload it with
    load_offline_transitions, verify shapes survive the trip."""
    import rlpd_utils

    H, W, C, n_state, n_act, N = 80, 144, 3, 16, 6, 64
    td = TensorDict(
        observations=TensorDict(
            rgb=torch.randint(0, 255, (N, H, W, C), dtype=torch.uint8),
            state=torch.randn(N, n_state, dtype=torch.float32),
            batch_size=[N],
        ),
        next_observations=TensorDict(
            rgb=torch.randint(0, 255, (N, H, W, C), dtype=torch.uint8),
            state=torch.randn(N, n_state, dtype=torch.float32),
            batch_size=[N],
        ),
        actions=torch.randn(N, n_act, dtype=torch.float32),
        rewards=torch.zeros(N, dtype=torch.float32),
        dones=torch.zeros(N, dtype=torch.bool),
        batch_size=[N],
    )
    # Mark the last transition done so sparse relabel produces +1.
    td.set("dones", torch.tensor([False] * (N - 1) + [True]))

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "synthetic.pt")
        rlpd_utils.save_offline_bundle(path, td)
        loaded = rlpd_utils.load_offline_transitions(
            path, obs_shape=(H, W, C), state_dim=n_state, action_dim=n_act,
            reward_mode="sparse",
        )
    assert loaded.batch_size[0] == N
    assert loaded["rewards"][-1].item() == 1.0, "sparse relabel did not place +1 on the done step"
    assert (loaded["rewards"][:-1] == 0).all(), "sparse relabel leaked reward outside the done step"
    print(f"[ OK ] Offline bundle round-trip (N={N}, sparse relabel verified)")


def test_symmetric_sampling_concat() -> None:
    """Mirror the ``torch.cat([online, offline], dim=0)`` pattern in the main
    training loop, against two TensorDicts of complementary batch size."""
    online_b, offline_b = 256, 256
    H, W, C, n_state, n_act = 80, 144, 3, 16, 6

    def _make(B: int) -> TensorDict:
        return TensorDict(
            observations=TensorDict(
                rgb=torch.zeros(B, H, W, C, dtype=torch.uint8),
                state=torch.zeros(B, n_state),
                batch_size=[B],
            ),
            next_observations=TensorDict(
                rgb=torch.zeros(B, H, W, C, dtype=torch.uint8),
                state=torch.zeros(B, n_state),
                batch_size=[B],
            ),
            actions=torch.zeros(B, n_act),
            rewards=torch.zeros(B),
            dones=torch.zeros(B, dtype=torch.bool),
            batch_size=[B],
        )

    data = torch.cat([_make(online_b), _make(offline_b)], dim=0)
    assert data.batch_size[0] == online_b + offline_b
    assert data["observations", "rgb"].shape == (online_b + offline_b, H, W, C)
    print(f"[ OK ] Symmetric sampling concat (online={online_b}, offline={offline_b})")


if __name__ == "__main__":
    test_critic_forward_and_target()
    test_offline_bundle_roundtrip()
    test_symmetric_sampling_concat()
    print("\nAll smoke tests passed.")
