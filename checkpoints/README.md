# `checkpoints/`

The checkpoint is not shipped with the code. Download it before running
`scripts/verify_benchmark.sh`:

```bash
hf download wxyxixixi/artic-o artic_o_s0_36.pth --local-dir ./checkpoints/
```

which gives:

```
checkpoints/
  artic_o_s0_36.pth      # 2.7 GB
```

The file holds `{"model": state_dict, "epoch": 36}`. Optimizer and scheduler
state have been stripped from the original training checkpoint — `eval.py`
reads only the `model` entry, so this has no effect on results.

Pass it with `--ckpt-path`; the config does not point at it
(`load_pretrain: false` in `src/configs/artic_o.yaml`).


