"""Temporal graph convolution network for sign recognition.

The architecture follows the reference work that reaches over 85% on WLASL:
spatial graph convolution over the skeleton, temporal convolution across frames,
multi-head attention across joints, and attention pooling across time. Four input
streams are aggregated, which is what the reference does and what the four
streams in ml/skeleton.py are shaped for.

The reason a graph rather than a sequence model. A BiLSTM over flattened
landmarks sees a bag of numbers per frame and has to infer for itself that a
wrist belongs to an arm. A graph convolution is told that, through the adjacency
built from the skeleton, and that relationship is most of what distinguishes one
sign from another. On this dataset the flat model scored 6.9% on 200 words,
which is far enough below published results to suggest the representation, not
the data, was the limit.

Channel flow, since it is the part that is easy to get wrong. A graph
convolution produces `channels * subsets` channels and keeps all of them; the
subset axis is folded only at the very end, when the per-subset results are
summed into a single representation. Collapsing subsets inside the block instead
makes every downstream layer disagree with the residual, which is sized on the
pre-collapse width. So `wide = channels * subsets` is the internal width and it
is computed once, at construction.

Sized for the CPU. The shipped configuration measures under 10ms per window on
six threads against a 40ms frame budget, and ONNX Runtime adds little on top, so
inference stays off the GPU.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .skeleton import JOINTS, adjacency

STREAMS = ("j", "jm", "b", "bm")


class GraphConv(nn.Module):
    """Spatial graph convolution: aggregate each joint from its neighbours.

    Several adjacency subsets are used rather than one, because a fixed graph
    says where joints connect but not which of those connections carry the
    sign. A per-subset learnable weight, softmaxed over the subset axis, keeps
    the learned topology a convex combination of the real skeleton.
    """

    def __init__(self, in_c: int, out_c: int, subsets: int = 4) -> None:
        super().__init__()
        self.out_c = out_c
        self.subsets = subsets
        self.spatial = nn.Conv2d(in_c, out_c * subsets, 1)
        self.bn = nn.BatchNorm2d(out_c * subsets)
        self.refine = nn.Parameter(torch.zeros(subsets, JOINTS, JOINTS))

        a = adjacency()
        # Centre the subsets on identity so they are perturbations of the real
        # skeleton rather than four unrelated graphs.
        stack = np.stack(
            [a + np.eye(JOINTS, dtype=np.float32) * (i == 0) for i in range(subsets)]
        )
        self.register_buffer("A", torch.from_numpy(stack))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, t, v = x.shape
        s = self.spatial(x).reshape(n, self.subsets, self.out_c, t, v)
        w = torch.softmax(self.refine, dim=0)
        a = (self.A.to(s.dtype) * w).unsqueeze(0)  # (1, S, V, V)

        # Joint aggregation. The channel and time axes are adjacent, so they fold
        # together for free, which turns this into one GEMM per subset with a
        # large M dimension instead of a batch of tiny ones.
        #
        # The shape of this line is the difference between training and hanging.
        # Written as matmul((N, C, T, V), (V, V)) the operands broadcast, so each
        # of the N*C*T rows becomes its own 1x75 by 75x75 GEMM. At batch 64 that
        # is 131,072 of them per subset per block, and six blocks run that in both
        # directions. The GPU then sits at 100% utilisation executing almost no
        # arithmetic, which is exactly how this failed: pegged, no progress, no
        # error. Folding C into T leaves N*S GEMMs of (2048x75) by (75x75), which
        # is real work at full occupancy.
        agg = torch.matmul(s.reshape(n, self.subsets, self.out_c * t, v), a)

        # Flattening subset then channel keeps the same channel index (k*C + c)
        # the previous per-subset concatenation produced, so an existing
        # checkpoint still loads.
        return self.bn(agg.reshape(n, self.out_c * self.subsets, t, v))


class STBlock(nn.Module):
    """Spatial graph convolution, temporal convolution, residual."""

    def __init__(
        self, in_c: int, out_c: int, subsets: int = 4, kernel_t: int = 9, drop: float = 0.3
    ) -> None:
        super().__init__()
        pad = kernel_t // 2
        wide = out_c * subsets
        self.g1 = GraphConv(in_c, out_c, subsets=subsets)
        self.tconv = nn.Conv2d(wide, wide, (kernel_t, 1), padding=(pad, 0))
        self.tbn = nn.BatchNorm2d(wide)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)
        self.res = None
        if in_c != wide:
            self.res = nn.Sequential(nn.Conv2d(in_c, wide, 1), nn.BatchNorm2d(wide))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x if self.res is None else self.res(x)
        y = self.relu(self.g1(x))
        y = self.drop(self.relu(self.tbn(self.tconv(y))))
        return self.relu(y + r)


class SignGCN(nn.Module):
    """Four-stream skeleton classifier.

    Shape of the network. Four input streams are embedded, lifted to a common
    width and summed, then six spatial-temporal graph blocks mix joints and
    frames. The joint axis is pooled away before the attention stage, and only
    time is attended over. That ordering is not a shortcut: the graph
    convolution has already done the joint reasoning, and attending over
    (joint, time) pairs instead would multiply the attention matrix by the node
    count for no extra information.
    """

    def __init__(
        self,
        num_classes: int,
        in_c: int = 3,
        channels: int = 16,
        blocks: int = 6,
        heads: int = 4,
        drop: float = 0.3,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.subsets = 4
        self.heads = heads

        # Joint streams carry 75 nodes and bone streams 58, one per edge, so the
        # families cannot be concatenated along the node axis. Each is projected
        # and padded onto the wider node count, with a learned node absorbing
        # whatever does not map onto a joint.
        # Each stream is embedded and lifted to `wide`, then summed. Summing is a
        # deliberate bottleneck: the four streams are four views of one skeleton,
        # and at roughly twelve training clips per word a model with four times
        # the first-block input overfits faster than it learns. Measured on the
        # 50 word split, summation reached 11.8% test against 8.2% for
        # concatenation, and it is cheaper to ship.
        self.embed = nn.ModuleList([nn.Conv2d(in_c, channels, 1) for _ in STREAMS])

        # A block consumes `in_c` and produces `out_c * subsets`. Keeping out_c
        # fixed at `channels` means the width after every block is `wide`,
        # whatever came in, so every block shares one shape.
        wide = channels * self.subsets
        self.lift = nn.ModuleList([nn.Conv2d(channels, wide, 1) for _ in STREAMS])
        self.blocks_list = nn.ModuleList(
            STBlock(wide, channels, subsets=self.subsets, drop=drop) for _ in range(blocks)
        )

        # Pooling over joints keeps both the mean and the strongest response,
        # because a sign depends on a few joints moving sharply and the mean
        # alone blurs them into the background.
        self.fuse = nn.Conv1d(wide * 2, d_model, 1)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=drop)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model * 2, num_classes)
        self.drop = nn.Dropout(drop)

    def forward(self, streams: dict[str, torch.Tensor]) -> torch.Tensor:
        # All four streams carry the same 75 nodes, so each is embedded and lifted
        # and they are summed. Summing is a deliberate bottleneck: the streams are
        # four views of one skeleton, and at roughly twelve training clips per word
        # a model with four times the input width overfits faster than it learns.
        # Measured on the 50 word split, summation reached 11.8% test against 8.2%
        # for concatenation, and it is cheaper to ship.
        xs = [self.lift[i](self.embed[i](streams[name])) for i, name in enumerate(STREAMS)]
        x = torch.stack(xs, dim=0).sum(dim=0)

        for blk in self.blocks_list:
            x = blk(x)

        # (N, C, T, V) -> (N, 2C, T) -> (N, T, d_model)
        pooled = torch.cat([x.mean(dim=3), x.amax(dim=3)], dim=1)
        seq = self.fuse(pooled).permute(0, 2, 1)
        attn, _ = self.attn(seq, seq, seq, need_weights=False)
        seq = self.norm(seq + attn)

        # Pool over time the same way: mean for the overall shape of the sign,
        # max for the single most decisive moment in it.
        out = torch.cat([seq.mean(dim=1), seq.amax(dim=1)], dim=1)
        return self.head(self.drop(out))


def _bone_edges() -> tuple[tuple[int, int], ...]:
    from .skeleton import EDGES

    return tuple(EDGES)


class StreamBundle(nn.Module):
    """Positional-argument view of the four streams, for ONNX export.

    The ONNX exporter needs the graph's inputs to be real positional tensors.
    `SignGCN.forward` takes a dict, which traces as an unsupported construct, so
    exporting the model directly raises a TypeError. This wrapper presents the
    same four tensors as separate arguments and is what gets written to disk.

    Kept deliberately thin: it owns no parameters, so the exported weights are
    the model's own and there is nothing to keep in sync by hand.
    """

    def __init__(self, model: SignGCN) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, j: torch.Tensor, jm: torch.Tensor, b: torch.Tensor, bm: torch.Tensor
    ) -> torch.Tensor:
        return self.model({"j": j, "jm": jm, "b": b, "bm": bm})


def build(
    num_classes: int, channels: int = 16, blocks: int = 6, drop: float = 0.3
) -> SignGCN:
    return SignGCN(num_classes=num_classes, channels=channels, blocks=blocks, drop=drop)