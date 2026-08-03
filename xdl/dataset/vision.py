"""Vision datasets backed by torchvision.

如需扩展，按相同模式添加即可。注册在 __init__.py 中完成。
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from torch.utils.data import Dataset

_import_error: Exception | None = None

try:
    import torchvision  # noqa: F401
except ImportError as exc:
    _import_error = exc


if _import_error is None:

    class _Wrapper(Dataset[Tuple[Any, Any]]):
        """torchvision 数据集统一包装。"""

        _dataset_cls: type

        def __init__(
            self,
            root: str = "./data",
            train: bool = True,
            download: bool = True,
            transform: Optional[Any] = None,
        ) -> None:
            self.dataset = self._dataset_cls(
                root=root, train=train, download=download, transform=transform
            )

        def __getitem__(self, idx: int) -> Tuple[Any, Any]:
            return self.dataset[idx]

        def __len__(self) -> int:
            return len(self.dataset)

    class CIFAR10Dataset(_Wrapper):
        _dataset_cls = torchvision.datasets.CIFAR10

    class MNISTDataset(_Wrapper):
        _dataset_cls = torchvision.datasets.MNIST

else:
    CIFAR10Dataset = None  # type: ignore[assignment,misc]
    MNISTDataset = None  # type: ignore[assignment,misc]
