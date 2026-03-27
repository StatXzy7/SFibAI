# Checkpoints

Place model weights required by local runs in this directory.

## Main SFibAI pipeline

Typical usage expects a checkpoint path such as:

```text
checkpoints/SFibAI.pth
```

## Lee VGG baseline

If ImageNet-pretrained VGG-16 weights are used for the reproduced Lee baseline, place them here, for example:

```text
checkpoints/vgg16-397923af.pth
```

## Public repository note

Large binary weights are not required to be committed to the public repository. This directory is kept so that command examples and relative paths remain consistent.
