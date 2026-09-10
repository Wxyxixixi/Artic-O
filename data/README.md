# `data/`

Nothing here is shipped with the code. Populate this directory before running
`scripts/verify_benchmark.sh`.

```
data/
  artic_o_test/                    # HuggingFace dataset split (~18 GB)
    dataset_dict.json
    test/
  val_views.json                   # deterministic per-sample view picks (~45 KB)
  larm_train_test_list.json        # train/test split definition
```

Paths are wired into `src/configs/artic_o.yaml`
(`hf_path`, `split_json`, `val_view_picks_path`). Change them there if you put
the assets elsewhere.

All three are required. `val_views.json` in particular is not optional: without
it the loader picks views at random per run, which adds roughly 3pp of variance
to the articulation metrics and makes the numbers non-reproducible.

`artic_o_test/` is written by `Dataset.save_to_disk`, so it must be opened with
`load_from_disk`; `load_dataset` will not work on it.

## Origin

These are derived from PartNet-Mobility and are subject to its terms; see
`THIRD_PARTY-NOTICES.md`.

Download: `hf download wxyxixixi/artic-o-data --repo-type dataset --local-dir ./data/`
