# Third-party components

The code in this repository is MIT licensed (see `LICENSE`). Two dependencies
it needs are **not** distributed here, because their licenses are incompatible
with MIT redistribution. `scripts/fetch_third_party.sh` downloads them from
upstream at pinned commits, and each remains under its own terms.

Running that script brings those terms onto your machine. **If you intend any
commercial use, read this file first** — the MIT license on this repository
does not, and cannot, relicense the code it fetches.

## Fetched, not vendored

### CroCo — `third_party/croco/`
- Copyright 2022-present NAVER Corp. · <https://github.com/naver/croco>
- **CC BY-NC-SA 4.0** — non-commercial, share-alike.
- Pinned commit `5d4dbc920b4cc0dac66bef0ce6876b58f1c82deb`.
- Used for exactly one symbol: `models/blocks.py::DecoderBlock`, imported by
  `src/models/heads/pts3d_decoder/flowm_decoder_point_joint_v2.py`. The bundled
  `curope` CUDA extension is not used and needs no compilation.
- The non-commercial and share-alike terms attach to CroCo's code, not to this
  repository's own source.

### TripoSG — `third_party/triposg/`
- Copyright (c) 2025 VAST-AI-Research and contributors ·
  <https://github.com/VAST-AI-Research/TripoSG>
- Derived from Tencent HunyuanDiT and distributed under the **Tencent Hunyuan
  Community License Agreement**, reproduced in the header of
  `triposg/models/transformers/triposg_transformer.py`. Note its territorial
  restriction — the agreement states it does not apply in the European Union.
- Pinned commit `fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c`.
- Used for `DiTBlock` and `FrequencyPositionalEmbedding`, imported by
  `src/models/heads/triposg_model/autoencoder_kl_triposg.py`. Upstream nests the
  package one level down, so the fetch script links
  `third_party/triposg -> _TripoSG/triposg`.

## Released weights

The checkpoint on HuggingFace is **not** covered by this repository's MIT
license. It is released for non-commercial research use — see the model card at
<https://huggingface.co/wxyxixixi/artic-o>. The architecture it instantiates
includes CroCo- and TripoSG-derived components, so the more restrictive
upstream terms are the safe reading for the trained weights.

## Derived implementations

Our own code, written against published methods. Provenance is noted in the
file headers so the lineage is auditable.

| Our file | Follows |
|---|---|
| `src/models/heads/seg_pat.py` | PARTICULATE's slot-attention part segmenter |
| `src/datasets/articulation.py` | LARM's articulation metric definitions |

The model architecture builds on **NOVA3R** (latent geometry encoder and
flow-matching point decoder). Weights released here were trained by us.

## Data

The test split derives from **PartNet-Mobility**
(<https://sapien.ucsd.edu/browse>) and remains subject to the PartNet-Mobility
terms of use. It is distributed separately at
<https://huggingface.co/datasets/wxyxixixi/artic-o-data>, not under this
repository's license. Obtain the original dataset from SAPIEN directly for any
use beyond reproducing our evaluation.

## Evaluation protocol

`docs/larm_eval.json` records which samples the **LARM** baseline evaluated and
whether it succeeded, so our tables use the same sample set. It contains only
sample identifiers and status flags — no LARM code or weights are redistributed
here.
