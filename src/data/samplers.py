"""Texture-grouped batch sampler ("PK sampler"): P distinct textures x K patches per batch.

Motivation (claude/whc2027_apt_relabeling_strategy.md, under-fitting / range-compression strategy,
"3+4" proposal): the default DataLoader(shuffle=True) draws patches i.i.d. from the whole train pool,
so a batch (size ~56) essentially never contains more than one patch of the same texture out of
~100 textures x ~64 patches. That makes any texture-level signal - an aggregated prediction, a
pairwise ranking comparison across textures - impossible to compute within a batch, which is exactly
what a texture-level aggregation loss and a pairwise-ranking auxiliary loss need (added separately on
top of this sampler). This sampler groups P distinct textures x K patches per batch instead, so every
batch always has several different textures' K-patch groups to work with.

Enable with --group_batch (mutually exclusive with --lds, which also replaces the default sampler -
see test/cv/cv_experiment.py / test/holdout/holdout_eval.py).
"""
import math
import random

import numpy as np


class TextureGroupedBatchSampler:
    """Batch sampler for torch.utils.data.DataLoader(dataset, batch_sampler=this).

    Each epoch: shuffle the texture order, split into chunks of P textures; for every texture in a
    chunk, draw K of its patch indices (without replacement if that texture has >= K patches this
    draw, otherwise with replacement so every included texture still contributes exactly K patches).
    Every yielded batch therefore has exactly len(chunk) * K samples. The last chunk of an epoch may
    hold fewer than P textures (if n_textures % P != 0), but every texture in it still contributes K
    patches - so __len__ is ceil(n_textures / P), not based on total patch count.

    texture_ids must be aligned index-for-index with the dataset this sampler is used on (e.g.
    all_df.texture.values[i_tr], matching the index order NormalizedSubset(base, i_tr, ...) wraps -
    see build_concat_base()'s callers in test/cv/cv_experiment.py and test/holdout/holdout_eval.py).
    """

    def __init__(self, texture_ids, P, K):
        self.P, self.K = int(P), int(K)
        self.by_tex = {}
        for idx, t in enumerate(np.asarray(texture_ids)):
            self.by_tex.setdefault(t, []).append(idx)
        self.textures = sorted(self.by_tex)
        if not self.textures:
            raise ValueError('TextureGroupedBatchSampler: no textures found in texture_ids')
        if self.P > len(self.textures):
            raise ValueError(
                f'--group_P {self.P} exceeds the number of training textures ({len(self.textures)})')
        if self.K < 1:
            raise ValueError(f'--group_K must be >= 1, got {self.K}')

    def __len__(self):
        return math.ceil(len(self.textures) / self.P)

    def __iter__(self):
        order = list(self.textures)
        random.shuffle(order)                                   # relies on the caller's global seeding (set_seed())
        for i in range(0, len(order), self.P):
            chunk = order[i:i + self.P]
            batch = []
            for t in chunk:
                pool = self.by_tex[t]
                batch.extend(random.sample(pool, self.K) if len(pool) >= self.K else random.choices(pool, k=self.K))
            random.shuffle(batch)                                # don't leave the K-per-texture grouping visible in batch order
            yield batch
