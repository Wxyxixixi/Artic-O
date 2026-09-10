#!/usr/bin/env bash
# Fetch the two upstream dependencies this repo does not vendor.
#
# They are kept out of the tree because they carry licenses incompatible with
# this repo's MIT license (CroCo is CC BY-NC-SA 4.0; TripoSG is under the
# Tencent Hunyuan Community License). Fetching them locally leaves each under
# its own terms — see THIRD_PARTY-NOTICES.md before using either commercially.
#
# Commits are pinned so the architecture matches the released checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

CROCO_URL=https://github.com/naver/croco.git
CROCO_COMMIT=5d4dbc920b4cc0dac66bef0ce6876b58f1c82deb

TRIPOSG_URL=https://github.com/VAST-AI-Research/TripoSG.git
TRIPOSG_COMMIT=fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c

# Shallow-fetch one pinned commit without downloading the whole history.
fetch_pinned () {
    local dst=$1 url=$2 commit=$3
    if [ -d "$dst/.git" ]; then
        echo "  $dst already present, skipping"
        return
    fi
    mkdir -p "$dst"
    git -C "$dst" init --quiet
    git -C "$dst" remote add origin "$url" 2>/dev/null || true
    git -C "$dst" fetch --quiet --depth 1 origin "$commit"
    git -C "$dst" checkout --quiet FETCH_HEAD
    echo "  $dst @ ${commit:0:10}"
}

mkdir -p third_party

# CroCo: repo root holds models/, so it maps straight onto the import path
# third_party.croco.models.blocks
echo "CroCo:"
fetch_pinned third_party/croco "$CROCO_URL" "$CROCO_COMMIT"

# TripoSG: upstream nests the package one level down (TripoSG/triposg/...),
# so link the inner package to the path we import, third_party.triposg
echo "TripoSG:"
fetch_pinned third_party/_TripoSG "$TRIPOSG_URL" "$TRIPOSG_COMMIT"
ln -sfn _TripoSG/triposg third_party/triposg
echo "  third_party/triposg -> _TripoSG/triposg"

echo
echo "Verifying imports..."
python -c "
from third_party.croco.models.blocks import DecoderBlock
from third_party.triposg.models.transformers.triposg_transformer import DiTBlock
from third_party.triposg.models.embeddings import FrequencyPositionalEmbedding
print('  OK: DecoderBlock, DiTBlock, FrequencyPositionalEmbedding')
"
