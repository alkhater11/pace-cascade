# Data

Run `python scripts/download_data.py` (or `--no-cifar` for the MLP/CNN studies only).
Files are verified against MD5 checksums (see the script).

| file | source | used by |
|---|---|---|
| `mn_*.gz` | MNIST, original IDX files (GitHub mirror `fgnt/mnist`) | MLP, SmallCNN |
| `fm_*.gz` | Fashion-MNIST, official repository `zalandoresearch/fashion-mnist` | MLP, SmallCNN, ViT-4 |
| `cifar10_jpgmirror.npz` | CIFAR-10 via the GitHub mirror `YoongiKim/CIFAR-10-images` (50,000 + 10,000 images stored as JPEG, IJG quality 75), decoded to uint8 | ResNet-8 |

**CIFAR-10 caveat.** The canonical CIFAR-10 host was not reachable from the build
environment, so the experiments used the JPEG mirror above. JPEG re-encoding
perturbs pixel values slightly, so absolute accuracies are not directly comparable
with published CIFAR-10 numbers; all methods in the paper are compared on the
identical data, which is what the study requires.
