"""
CV-Scope Reporting Module.

Accumulates pipeline execution metrics and produces a structured, machine-readable
JSON report containing verified numerical results, statistical distributions,
parameters, and processing timestamps.
"""

import json
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np

from src.utils import ensure_dir


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to safely serialize NumPy scalar types and arrays."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


class AnalysisReport:
    """Accumulates diagnostic parameters, module metrics, and execution times."""

    def __init__(self, mode: str, input_files: Dict[str, str]):
        self.report_data: Dict[str, Any] = {
            "project": "CV-Scope: Computer Vision Scene Geometry and Depth Analysis System",
            "version": "1.0.0",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "mode": mode,
            "inputs": input_files,
            "algorithms_applied": [],
            "processing_times_ms": {},
            "metrics": {},
            "warnings": [],
            "status": "INITIALIZED"
        }

    def add_algorithm(self, name: str) -> None:
        """Register an algorithm as part of the executed pipeline."""
        if name not in self.report_data["algorithms_applied"]:
            self.report_data["algorithms_applied"].append(name)

    def record_time(self, module_name: str, elapsed_ms: float) -> None:
        """Record execution latency for a pipeline stage."""
        self.report_data["processing_times_ms"][module_name] = round(elapsed_ms, 2)

    def update_metrics(self, section: str, metrics: Dict[str, Any]) -> None:
        """Add or update metrics for a specific analysis component."""
        self.report_data["metrics"][section] = metrics

    def add_warning(self, message: str) -> None:
        """Append a non-fatal warning message."""
        self.report_data["warnings"].append(message)

    def finalize(self, success: bool = True) -> Dict[str, Any]:
        """Compute aggregate time and mark execution status."""
        self.report_data["status"] = "SUCCESS" if success else "FAILED"
        total_time = sum(self.report_data["processing_times_ms"].values())
        self.report_data["total_processing_time_ms"] = round(total_time, 2)
        return self.report_data

    def save_json(self, output_path: Union[str, Path]) -> str:
        """Save formatted JSON report to disk."""
        p = Path(output_path)
        ensure_dir(p.parent)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.report_data, f, indent=4, cls=NumpyJSONEncoder)
        return str(p.resolve())
