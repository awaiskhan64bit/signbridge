"""Tests for the skeleton graph representation and the model built on it.

Three of these exist because the corresponding bug was actually hit during
development rather than imagined:

- the rotation test, because `augment` wrote x and then read x back, rotating
  the coordinates twice
- the graph-conv equivalence test, because folding the wrong pair of axes made
  the aggregation run as 131,072 tiny matmuls and pegged the GPU at 100%
- the connectivity test, because the hand subtrees had no edge to the pose and
  the convolution could not move information between them
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from signbridge.ml import skeleton as sk  # noqa: E402
from signbridge.ml.gcn import GraphConv, StreamBundle, build  # noqa: E402
from signbridge.ml.holistic import L_HIP, L_SHOULDER, POSE, R_HIP, R_SHOULDER  # noqa: E402


def _component_sizes() -> list[int]:
    """Sizes of the connected components of the adjacency graph."""

    a = sk.adjacency() > 0
    seen = np.zeros(sk.JOINTS, dtype=bool)
    sizes = []
    for start in range(sk.JOINTS):
        if seen[start]:
            continue
        stack, size = [start], 0
        seen[start] = True
        while stack:
            node = stack.pop()
            size += 1
            for nxt in np.flatnonzero(a[node] & ~seen):
                seen[nxt] = True
                stack.append(int(nxt))
        sizes.append(size)
    return sorted(sizes)


class TestGraph:
    def test_adjacency_is_symmetric_with_self_loops(self):
        a = sk.adjacency()
        assert a.shape == (sk.JOINTS, sk.JOINTS)
        np.testing.assert_allclose(a, a.T, atol=1e-6)
        assert (np.diag(a) > 0).all()

    def test_adjacency_is_finite_and_bounded(self):
        a = sk.adjacency()
        assert np.isfinite(a).all()
        # Self-loops guarantee a non-zero degree, so the normalised entries stay
        # small and bounded. An uninitialised entry from a masked power shows up
        # here as a large or NaN value.
        assert a.min() >= 0.0 and a.max() <= 1.0

    def test_no_isolated_nodes(self):
        deg = (sk.adjacency() > 0).sum(axis=1)
        assert (deg >= 2).all(), f"joints with too few neighbours: {np.flatnonzero(deg < 2)}"

    def test_every_joint_is_reachable_from_every_other(self):
        # The single most important structural property. A second component means
        # pose cannot inform the hands, which is the whole reason pose is here.
        assert _component_sizes() == [sk.JOINTS]

    def test_hands_are_attached_to_the_pose(self):
        adj = sk.adjacency() > 0
        pose = slice(sk.POSE_START, sk.JOINTS)
        assert adj[: sk.POSE_START, pose].any(), "no hand joint reaches the pose"

    def test_edges_reference_real_joints(self):
        assert all(0 <= a < sk.JOINTS and 0 <= b < sk.JOINTS for a, b in sk.EDGES)


class TestNormalisation:
    def _clip(self, frames: int = 20) -> np.ndarray:
        rng = np.random.default_rng(0)
        clip = rng.normal(0.3, 0.05, size=(frames, 553, 3)).astype(np.float32)
        # Plant a plausible body so the frame can be solved.
        pose = clip[:, POSE, :]
        pose[:, L_SHOULDER] = (-0.4, 0.0, 0.0)
        pose[:, R_SHOULDER] = (0.4, 0.0, 0.0)
        pose[:, L_HIP] = (-0.3, 1.0, 0.0)
        pose[:, R_HIP] = (0.3, 1.0, 0.0)
        return clip

    def test_scale_and_offset_cancel(self):
        clip = self._clip()
        base = sk.build_input(clip, 32)
        moved = sk.build_input(clip * 3.0 + 7.0, 32)
        np.testing.assert_allclose(base, moved, atol=1e-3)

    def test_output_shape_is_window_by_joints_by_three(self):
        assert sk.build_input(self._clip(40), 64).shape == (64, sk.JOINTS, 3)

    def test_a_global_rotation_is_cancelled(self):
        # Camera roll is the reason the body frame exists. Rotating the whole
        # skeleton and then expressing it relative to the shoulders must give back
        # the same numbers, otherwise a signer holding the phone at an angle is a
        # different class from one holding it level.
        clip = self._clip()
        ang = 0.3
        row = np.array(
            [[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]], dtype=np.float32
        )
        turned = clip.copy()
        turned[..., :2] = clip[..., :2] @ row
        np.testing.assert_allclose(
            sk.build_input(clip, 32), sk.build_input(turned, 32), atol=1e-3
        )

    def test_moving_a_hand_relative_to_the_body_is_not_cancelled(self):
        # The converse, and the reason to keep the body at all: the same hand
        # shape at the forehead and at the chest has to encode differently.
        clip = self._clip()
        raised = clip.copy()
        raised[:, 0:21, 1] -= 0.8  # lift the left hand well above the shoulders
        assert not np.allclose(
            sk.build_input(clip, 32), sk.build_input(raised, 32), atol=1e-2
        )


class TestStreams:
    def test_all_streams_share_one_node_count(self):
        # Bones are one vector per joint, not one node per bone, so every stream
        # has exactly JOINTS nodes and nothing needs padding. If this drifts, the
        # model and its adjacency disagree about the node axis again.
        clip = np.zeros((8, 553, 3), dtype=np.float32)
        streams = sk.four_streams(clip, 16)
        assert {v.shape[-2] for v in streams.values()} == {sk.JOINTS}

    def test_bone_is_the_joint_minus_its_parent(self):
        j = np.random.default_rng(1).normal(size=(4, sk.JOINTS, 3)).astype(np.float32)
        b = sk.bone_stream(j)
        child = np.flatnonzero(~sk._IS_ROOT)
        np.testing.assert_allclose(
            b[..., child, :], j[..., child, :] - j[..., sk.PARENT[child], :], atol=1e-6
        )

    def test_roots_have_a_zero_bone(self):
        # A root has no parent, and zero is the convention already used for a
        # joint the tracker did not find.
        j = np.random.default_rng(1).normal(size=(4, sk.JOINTS, 3)).astype(np.float32)
        np.testing.assert_array_equal(sk.bone_stream(j)[:, sk._IS_ROOT], 0.0)

    def test_bone_stream_keeps_the_track_shape(self):
        j = np.zeros((6, sk.JOINTS, 3), dtype=np.float32)
        assert sk.bone_stream(j).shape == j.shape

    def test_there_is_a_parent_for_every_non_root(self):
        assert sk.PARENT.shape == (sk.JOINTS,)
        non_root = np.flatnonzero(~sk._IS_ROOT)
        assert (sk.PARENT[non_root] >= 0).all()
        assert (sk.PARENT[non_root] < sk.JOINTS).all()
        # A parent must come earlier in the layout or the tree is not a tree.
        assert (sk.PARENT[non_root] < non_root).all(), "a parent index follows its child"

    def test_motion_starts_at_zero(self):
        j = np.random.default_rng(2).normal(size=(6, sk.JOINTS, 3)).astype(np.float32)
        np.testing.assert_array_equal(sk.motion(j)[0], np.zeros_like(j[0]))

    def test_motion_on_a_batch_differences_time_not_clips(self):
        # The bug this guards: differencing axis 0 of a (N, T, V, 3) batch
        # subtracts one clip from the next and trains to nothing.
        batch = np.random.default_rng(3).normal(size=(5, 7, sk.JOINTS, 3)).astype(np.float32)
        got = sk.motion(batch)
        for n in range(5):
            np.testing.assert_allclose(got[n, 1:], batch[n, 1:] - batch[n, :-1], atol=1e-6)

    def test_motion_batched_matches_single_track(self):
        track = np.random.default_rng(4).normal(size=(7, sk.JOINTS, 3)).astype(np.float32)
        np.testing.assert_allclose(
            sk.motion(track), sk.motion(track[None])[0], atol=1e-5
        )

    def test_four_streams_are_all_present_and_finite(self):
        clip = np.zeros((12, 553, 3), dtype=np.float32)
        streams = sk.four_streams(clip, 16)
        assert set(streams) == {"j", "jm", "b", "bm"}
        assert all(np.isfinite(v).all() for v in streams.values())


class TestGraphConv:
    def test_matches_the_per_subset_concatenation(self):
        # The fast form must be the same function as the obvious one, or the
        # speedup bought nothing but a different model.
        torch.manual_seed(0)
        g = GraphConv(8, 6, subsets=4).eval()
        x = torch.randn(3, 8, 16, sk.JOINTS)
        with torch.no_grad():
            a = g.A * torch.softmax(g.refine, dim=0)
            s = g.spatial(x).reshape(3, 4, 6, 16, sk.JOINTS)
            slow = torch.cat([torch.matmul(s[:, k], a[k]) for k in range(4)], dim=1)
            fast = torch.matmul(
                s.reshape(3, 4, 6 * 16, sk.JOINTS), a.unsqueeze(0)
            ).reshape(3, 24, 16, sk.JOINTS)
        torch.testing.assert_close(slow, fast, rtol=1e-4, atol=1e-4)

    def test_output_width_is_out_c_times_subsets(self):
        g = GraphConv(8, 6, subsets=4).eval()
        assert g(torch.randn(2, 8, 12, sk.JOINTS)).shape == (2, 24, 12, sk.JOINTS)

    def test_gradients_reach_every_parameter(self):
        m = build(num_classes=7, channels=8, blocks=2).train()
        streams = {name: torch.randn(2, 3, 16, sk.JOINTS) for name in ("j", "jm", "b", "bm")}
        m(streams).square().mean().backward()
        missing = [n for n, p in m.named_parameters() if p.grad is None]
        assert not missing, f"no gradient for {missing}"


def _batch(n: int = 2, t: int = 32):
    """One batch of the four streams, all on the shared JOINTS node axis."""

    return {name: torch.randn(n, 3, t, sk.JOINTS) for name in ("j", "jm", "b", "bm")}


class TestModel:

    def test_forward_gives_one_logit_per_class(self):
        m = build(num_classes=5, channels=8, blocks=2).eval()
        with torch.no_grad():
            assert m(_batch()).shape == (2, 5)

    def test_model_node_count_matches_the_graph(self):
        # The model convolves over the adjacency's node axis. If they disagree the
        # graph convolution raises on shapes that look fine in isolation.
        m = build(num_classes=3, channels=8, blocks=1).eval()
        with torch.no_grad():
            assert m(_batch()).shape == (2, 3)

    def test_output_is_finite(self):
        m = build(num_classes=5, channels=8, blocks=2).eval()
        with torch.no_grad():
            assert torch.isfinite(m(_batch())).all()

    def test_exports_to_onnx_and_matches_torch(self):
        # The shipped app runs ONNX Runtime on the CPU, so an export failure here
        # is a shipping failure, not a test artefact. It exports the bundle, not
        # the model, because the exporter needs positional inputs and the model's
        # forward takes a dict. The numerical comparison is the part that matters:
        # an exported graph that silently disagrees with its weights ships a model
        # that recognises nothing and reports no error.
        import onnxruntime as ort

        torch.manual_seed(0)
        m = build(num_classes=4, channels=8, blocks=2).eval()
        batch = _batch(1)
        with torch.no_grad():
            expected = m(batch)
        path = Path("tmp_gcn_test.onnx")
        try:
            torch.onnx.export(
                StreamBundle(m),
                tuple(batch.values()),
                str(path),
                input_names=["j", "jm", "b", "bm"],
                output_names=["logits"],
                opset_version=17,
                dynamo=False,
            )
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            assert sess.get_providers() == ["CPUExecutionProvider"]
            got = sess.run(None, {k: v.numpy() for k, v in batch.items()})[0]
        finally:
            path.unlink(missing_ok=True)
        torch.testing.assert_close(
            torch.from_numpy(got), expected, atol=1e-4, rtol=1e-4
        )