# Model checkpoints

Put the trained PyTorch checkpoint here, for example `models/final_model_5pct_rate.pth`,
and pass it to the generator:

```bash
resnet8-hls models/final_model_5pct_rate.pth
```

Checkpoint files (`*.pth`, `*.pt`, `*.ckpt`) are ignored by git because they are large
binary files. To share a checkpoint, attach it to a GitHub release or track it with
[Git LFS](https://git-lfs.com/).

See "Checkpoint format" in the top-level `README.md` for the supported formats. If you
only want to try the tool, create a randomly initialised checkpoint with:

```bash
python examples/make_demo_checkpoint.py models/demo.pth
```
