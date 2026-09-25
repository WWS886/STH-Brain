"""Train-only, subject-balanced frozen ROI normalization for STH-Brain TS-JEPA."""
from __future__ import annotations

from utils.artifact_schema import schema_matches
import hashlib
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np


NORMALIZATION_SCHEMA = "train_roi_frozen_normalization"
NORMALIZATION_ESTIMATOR = (
    "median_subject_scan_medians_and_1.4826_median_subject_scan_mads"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _uid_sha256(values) -> str:
    payload = json.dumps(sorted(str(value) for value in values), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _nanmedian(values: np.ndarray, axis: int) -> np.ndarray:
    # A subject may have an ROI masked in every scan; the cross-subject stage
    # still validates that each ROI has at least one finite train contribution.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def load_normalization_stats(
    path: str | Path,
    expected_sha256: str,
    *,
    expected_n_roi: int = 200,
) -> dict:
    """Load and validate the exact frozen-statistics bytes used by the model."""
    stats_path = Path(path).expanduser().resolve()
    payload = stats_path.read_bytes()
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != str(expected_sha256).strip().lower():
        raise ValueError("Frozen ROI normalization statistics SHA-256 mismatch")
    stats = json.loads(payload)
    required = {
        "schema_version",
        "estimator",
        "normalization_source",
        "manifest_sha256",
        "qc_receipt_sha256",
        "split_receipt_sha256",
        "sequence_cache_receipt_sha256",
        "fit_split_col",
        "fit_split",
        "subject_count",
        "scan_count",
        "subject_uid_sha256",
        "scan_uid_sha256",
        "n_roi",
        "scale_floor",
        "roi_subject_counts",
        "location",
        "scale",
    }
    if set(stats) != required:
        raise ValueError("Frozen ROI normalization statistics fields are invalid")
    if (
        stats["schema_version"] != NORMALIZATION_SCHEMA
        or not schema_matches(stats["estimator"], NORMALIZATION_ESTIMATOR)
        or stats["normalization_source"] != "raw_before_normalization"
        or int(stats["n_roi"]) != int(expected_n_roi)
        or stats["fit_split"] != "train"
    ):
        raise ValueError("Frozen ROI normalization statistics semantics are invalid")
    location = np.asarray(stats["location"], dtype=np.float64)
    scale = np.asarray(stats["scale"], dtype=np.float64)
    roi_subject_counts = np.asarray(stats["roi_subject_counts"], dtype=np.int64)
    expected_shape = (int(expected_n_roi),)
    if (
        location.shape != expected_shape
        or scale.shape != expected_shape
        or roi_subject_counts.shape != expected_shape
        or not np.isfinite(location).all()
        or not np.isfinite(scale).all()
        or not (scale > 0).all()
        or not (roi_subject_counts > 0).all()
        or not np.isfinite(float(stats["scale_floor"]))
        or float(stats["scale_floor"]) <= 0
        or int(stats["subject_count"]) < 1
        or int(stats["scan_count"]) < int(stats["subject_count"])
    ):
        raise ValueError("Frozen ROI normalization statistics values are invalid")
    return stats


def validate_normalization_provenance(stats: dict, cfg: dict) -> dict:
    """Bind fitted statistics to the configured train role and input receipts."""
    data = cfg["data"]
    expected = {
        "manifest_sha256": sha256_file(data["manifest"]),
        "qc_receipt_sha256": sha256_file(data["qc_receipt_path"]),
        "split_receipt_sha256": sha256_file(data["split_receipt_path"]),
        "sequence_cache_receipt_sha256": sha256_file(
            data["sequence_cache_receipt_path"]
        ),
        "fit_split_col": str(data["split_col"]),
        "fit_split": str(data["train_split"]),
    }
    mismatched = [key for key, value in expected.items() if stats.get(key) != value]
    if mismatched or expected["fit_split"] != "train":
        raise ValueError(
            "Frozen ROI normalization provenance disagrees with the configured "
            f"train inputs: fields={mismatched}"
        )
    return {
        "schema_version": NORMALIZATION_SCHEMA,
        "estimator": NORMALIZATION_ESTIMATOR,
        "fit_split_col": stats["fit_split_col"],
        "fit_split": stats["fit_split"],
        "subject_count": int(stats["subject_count"]),
        "scan_count": int(stats["scan_count"]),
        "subject_uid_sha256": stats["subject_uid_sha256"],
        "scan_uid_sha256": stats["scan_uid_sha256"],
    }


def fit_train_roi_statistics(dataset, cfg: dict) -> dict:
    """Fit one equally weighted robust location/scale contribution per subject."""
    data = cfg["data"]
    split_col = str(data["split_col"])
    train_split = str(data["train_split"])
    if train_split != "train" or split_col not in dataset.df.columns:
        raise ValueError("Frozen ROI normalization must be fitted on the STH-Brain train role")
    if not dataset.df[split_col].astype(str).eq(train_split).all():
        raise ValueError("Frozen ROI normalization received a non-train scan")

    locations_by_subject = defaultdict(list)
    scales_by_subject = defaultdict(list)
    scan_uids = []
    for index in range(len(dataset)):
        sample = dataset[index]
        values = sample["x"].detach().cpu().numpy().astype(np.float64, copy=False)
        valid_roi = sample["roi_valid_mask"].detach().cpu().numpy().astype(bool)
        scan_location = np.median(values, axis=0)
        scan_scale = 1.4826 * np.median(
            np.abs(values - scan_location[None, :]), axis=0
        )
        scan_location[~valid_roi] = np.nan
        scan_scale[~valid_roi] = np.nan
        subject_uid = str(sample["subject_uid"])
        locations_by_subject[subject_uid].append(scan_location)
        scales_by_subject[subject_uid].append(scan_scale)
        scan_uids.append(str(sample["scan_uid"]))

    subject_uids = sorted(locations_by_subject)
    subject_locations = np.stack(
        [_nanmedian(np.stack(locations_by_subject[uid]), axis=0) for uid in subject_uids]
    )
    subject_scales = np.stack(
        [_nanmedian(np.stack(scales_by_subject[uid]), axis=0) for uid in subject_uids]
    )
    roi_subject_counts = np.isfinite(subject_locations).sum(axis=0)
    location = _nanmedian(subject_locations, axis=0)
    robust_scale = _nanmedian(subject_scales, axis=0)
    scale_floor = float(cfg["model"]["block_normalization_epsilon"])
    scale = np.maximum(robust_scale, scale_floor)
    if (
        location.shape != (dataset.n_roi,)
        or not np.isfinite(location).all()
        or not np.isfinite(scale).all()
        or not (scale > 0).all()
        or not (roi_subject_counts > 0).all()
    ):
        raise RuntimeError("Train role cannot identify finite statistics for every ROI")

    return {
        "schema_version": NORMALIZATION_SCHEMA,
        "estimator": NORMALIZATION_ESTIMATOR,
        "normalization_source": "raw_before_normalization",
        "manifest_sha256": sha256_file(data["manifest"]),
        "qc_receipt_sha256": sha256_file(data["qc_receipt_path"]),
        "split_receipt_sha256": sha256_file(data["split_receipt_path"]),
        "sequence_cache_receipt_sha256": sha256_file(
            data["sequence_cache_receipt_path"]
        ),
        "fit_split_col": split_col,
        "fit_split": train_split,
        "subject_count": len(subject_uids),
        "scan_count": len(scan_uids),
        "subject_uid_sha256": _uid_sha256(subject_uids),
        "scan_uid_sha256": _uid_sha256(scan_uids),
        "n_roi": int(dataset.n_roi),
        "scale_floor": scale_floor,
        "roi_subject_counts": roi_subject_counts.astype(int).tolist(),
        "location": location.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
    }








