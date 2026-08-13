"""xdl-jax public API."""

from ._version import __version__
from .artifact import (
    ModelArtifact,
    ModelArtifactReport,
    export_model_artifact,
    materialize_model_artifact,
    read_model_artifact_metadata,
)
from .callbacks import (
    Callback,
    CallbackList,
    CheckpointCallback,
    CompileReportCallback,
    ConsoleLoggerCallback,
    EarlyStoppingCallback,
    TimerCallback,
)
from .checkpoint import (
    CheckpointMetadata,
    CheckpointReport,
    JaxCheckpointManager,
    RestoredCheckpoint,
)
from .data import ArrayDataSource, BatchSpec, JaxDataSource, validate_batch
from .distributed import (
    DataParallelStrategy,
    DeviceReport,
    JaxStrategy,
    MeshConfig,
    SingleDeviceStrategy,
)
from .errors import (
    CallbackError,
    CheckpointError,
    ConfigurationError,
    DataValidationError,
    TrainingError,
    XdlJaxError,
)
from .model import FunctionalModelAdapter, ModelAdapter, NNXModelAdapter
from .optimizer import build_optimizer
from .performance import BenchmarkReport, benchmark_callable
from .task import JaxTask
from .trainer import JaxTrainer, TrainerConfig
from .types import (
    JaxTrainerState,
    JaxTrainState,
    MetricSnapshot,
    ModelState,
    StepOutput,
    TrainResult,
)

__all__ = [
    "ArrayDataSource",
    "__version__",
    "ModelArtifact",
    "ModelArtifactReport",
    "BatchSpec",
    "Callback",
    "CallbackError",
    "CallbackList",
    "CheckpointCallback",
    "CompileReportCallback",
    "CheckpointError",
    "CheckpointMetadata",
    "CheckpointReport",
    "ConfigurationError",
    "ConsoleLoggerCallback",
    "DataValidationError",
    "DataParallelStrategy",
    "DeviceReport",
    "EarlyStoppingCallback",
    "FunctionalModelAdapter",
    "JaxCheckpointManager",
    "JaxTask",
    "JaxStrategy",
    "JaxTrainState",
    "JaxTrainer",
    "JaxTrainerState",
    "JaxDataSource",
    "MetricSnapshot",
    "ModelAdapter",
    "ModelState",
    "MeshConfig",
    "NNXModelAdapter",
    "RestoredCheckpoint",
    "SingleDeviceStrategy",
    "StepOutput",
    "TimerCallback",
    "TrainResult",
    "TrainingError",
    "XdlJaxError",
    "TrainerConfig",
    "build_optimizer",
    "BenchmarkReport",
    "benchmark_callable",
    "export_model_artifact",
    "materialize_model_artifact",
    "read_model_artifact_metadata",
    "validate_batch",
]
