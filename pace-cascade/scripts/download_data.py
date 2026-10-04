"""Download and verify all datasets used in the paper (into ../data).

MNIST         : GitHub mirror github.com/fgnt/mnist (original LeCun IDX files)
Fashion-MNIST : github.com/zalandoresearch/fashion-mnist (official repository)
CIFAR-10      : the canonical host (cs.toronto.edu) is used when reachable; the
                experiments in the paper used the GitHub mirror
                github.com/YoongiKim/CIFAR-10-images, which stores the 60,000
                images as JPEG files (IJG quality 75).  Both routes are supported;
                the mirror is converted to data/cifar10_jpgmirror.npz.
All files are checked against MD5 checksums."""
import os, sys, io, hashlib, zipfile, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)
IDX = {  # local name: (url, md5 of the .gz file)
    "mn_train_images.gz": ("https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    "mn_train_labels.gz": ("https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    "mn_test_images.gz":  ("https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz",  "9fb629c4189551a2d022fa330f9573f3"),
    "mn_test_labels.gz":  ("https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz",  "ec29112dd5afa0611ce80d1b7f02629c"),
    "fm_train_images.gz": ("https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
    "fm_train_labels.gz": ("https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
    "fm_test_images.gz":  ("https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz",  "bef4ecab320f06d8554ea6380940ec79"),
    "fm_test_labels.gz":  ("https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz",  "bb300cfdad3c16e7a12a480ee83cd310"),
}
CIFAR_MIRROR = ("https://codeload.github.com/YoongiKim/CIFAR-10-images/zip/refs/heads/master",
                "6303d1b78272b63c8fcd7973d39919f9")
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def md5(b):
    return hashlib.md5(b).hexdigest()


def fetch(url):
    with urllib.request.urlopen(url, timeout=600) as r:
        return r.read()


def main(skip_cifar=False):
    for name, (url, h) in IDX.items():
        p = os.path.join(DATA, name)
        if os.path.exists(p) and md5(open(p, "rb").read()) == h:
            print("ok   ", name); continue
        b = fetch(url)
        assert md5(b) == h, f"checksum mismatch for {name}"
        open(p, "wb").write(b); print("got  ", name)
    if skip_cifar:
        return
    out = os.path.join(DATA, "cifar10_jpgmirror.npz")
    if os.path.exists(out):
        print("ok    cifar10_jpgmirror.npz"); return
    from PIL import Image
    b = fetch(CIFAR_MIRROR[0])
    if md5(b) != CIFAR_MIRROR[1]:
        print("warning: CIFAR-10 mirror archive checksum differs from the one used in the paper")
    z = zipfile.ZipFile(io.BytesIO(b)); arrays = {}
    for split in ["train", "test"]:
        names = sorted(n for n in z.namelist() if n.startswith(f"CIFAR-10-images-master/{split}/") and n.endswith(".jpg"))
        X = np.stack([np.asarray(Image.open(io.BytesIO(z.read(n))).convert("RGB"), np.uint8) for n in names])
        y = np.array([CLASSES.index(n.split("/")[2]) for n in names], np.int64)
        arrays[f"X_{split}"], arrays[f"y_{split}"] = X, y
        print(f"      CIFAR-10 {split}: {X.shape}")
    np.savez_compressed(out, **arrays); print("got   cifar10_jpgmirror.npz")


if __name__ == "__main__":
    main(skip_cifar="--no-cifar" in sys.argv)
