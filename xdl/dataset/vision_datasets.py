"""视觉数据集模板 — torchvision 封装。

如需扩展，按相同模式添加即可。注册在 __init__.py 中完成。
"""

from xdl.utils.registry import DATASET_REGISTRY  # noqa: F401

_import_error: Exception | None = None

try:
    import torchvision  # noqa: F401
except ImportError as exc:
    _import_error = exc


if _import_error is None:

    class _Wrapper:
        """torchvision 数据集统一包装。"""
        def __init__(self, root="./data", train=True, download=True, transform=None):
            self.dataset = self._dataset_cls(
                root=root, train=train, download=download, transform=transform
            )

        def __getitem__(self, idx):
            return self.dataset[idx]

        def __len__(self):
            return len(self.dataset)

    class CIFAR10Dataset(_Wrapper):
        _dataset_cls = torchvision.datasets.CIFAR10

    class MNISTDataset(_Wrapper):
        _dataset_cls = torchvision.datasets.MNIST

else:
    CIFAR10Dataset = None  # type: ignore
    MNISTDataset = None  # type: ignore
