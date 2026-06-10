# Checkpoints

Place model weights required by local runs in this directory.

## Main SFibAI pipeline

Training does not require a checkpoint by default. Use `--init_checkpoint` only when initializing the backbone from a local pretrained checkpoint.

Evaluation and manuscript-scale figure reproduction require a trained checkpoint supplied by the user, typically:

```text
checkpoints/SFibAI.pth
```

The manuscript checkpoint is not redistributed in this public repository.

## Lee VGG baseline

If ImageNet-pretrained VGG-16 weights are used for the reproduced Lee baseline, place them here, for example:

```text
checkpoints/vgg16-397923af.pth
```
