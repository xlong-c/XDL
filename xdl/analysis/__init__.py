"""Model interpretation and representation analysis helpers for XDL."""

from .activation import ActivationCapture, ActivationRecord, capture_activations
from .attention import AttentionCapture, attention_rollout, attention_rollout_for_model, capture_attention_maps
from .concept import (
    ConceptProbe,
    concept_activation_vector,
    compute_module_tcav,
    directional_derivative,
    fit_concept_probe,
    tcav_from_probe,
    tcav_score,
)
from .grad_cam import GradCAM, compute_grad_cam
from .probe import LinearProbe, fit_linear_probe, score_linear_probe
from .report import (
    flatten_mapping,
    normalize_record,
    records_to_rows,
    write_analysis_bundle,
    write_csv_report,
    write_json_report,
    write_markdown_summary,
)

__all__ = [
    "ActivationCapture",
    "ActivationRecord",
    "AttentionCapture",
    "ConceptProbe",
    "GradCAM",
    "LinearProbe",
    "attention_rollout",
    "attention_rollout_for_model",
    "capture_activations",
    "capture_attention_maps",
    "concept_activation_vector",
    "compute_module_tcav",
    "compute_grad_cam",
    "directional_derivative",
    "fit_concept_probe",
    "fit_linear_probe",
    "flatten_mapping",
    "normalize_record",
    "records_to_rows",
    "score_linear_probe",
    "tcav_from_probe",
    "tcav_score",
    "write_analysis_bundle",
    "write_csv_report",
    "write_json_report",
    "write_markdown_summary",
]
