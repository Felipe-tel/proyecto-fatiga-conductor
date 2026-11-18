from pathlib import Path
import csv
from collections import defaultdict

import numpy as np

BASE_DIR = Path(__file__).resolve().parent

FRAMES_CSV = BASE_DIR / "data" / "features" / "yawdd_mar_frames.csv"
SUMMARY_CSV = BASE_DIR / "data" / "features" / "yawdd_mar_summary.csv"

SWEEP_CSV = (
    BASE_DIR
    / "data"
    / "features"
    / "yawdd_mar_normalized_cv_sweep.csv"
)

PREDICTIONS_CSV = (
    BASE_DIR
    / "data"
    / "features"
    / "yawdd_mar_normalized_validation_predictions.csv"
)

DEV_SUBJECTS = [
    "F01",
    "F02",
    "F06",
    "M04",
    "M05",
    "M06",
]

# Ya observado en iteraciones previas: usar como VALIDACION,
# no como test final.
VALIDATION_SUBJECTS = [
    "F09",
    "F10",
    "M09",
    "M15",
]


def subject_id_from_filename(filename: str) -> str:
    number = int(filename.split("-")[0])

    if "Female" in filename:
        return f"F{number:02d}"

    if "Male" in filename:
        return f"M{number:02d}"

    return f"U{number:02d}"


def binary_label(label: str) -> str:
    return "yawning" if label == "yawning" else "no_yawn"


def load_summary():
    rows = []

    with SUMMARY_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            video = row["video"]
            subject_id = subject_id_from_filename(video)

            if subject_id in DEV_SUBJECTS:
                split = "dev"
            elif subject_id in VALIDATION_SUBJECTS:
                split = "validation"
            else:
                split = "other"

            rows.append(
                {
                    "video": video,
                    "original_label": row["label"],
                    "binary_label": binary_label(row["label"]),
                    "subject_id": subject_id,
                    "split": split,
                    "fps": float(row["fps"]),
                }
            )

    return rows


def load_frames():
    by_video = defaultdict(list)

    with FRAMES_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            by_video[row["video"]].append(
                (
                    int(row["frame"]),
                    float(row["mar"]),
                )
            )

    for video in by_video:
        by_video[video].sort(
            key=lambda item: item[0]
        )

    return dict(by_video)


def smooth_sequence(frame_values, fps, window_seconds=0.20):
    if not frame_values:
        return []

    frames = np.asarray(
        [item[0] for item in frame_values],
        dtype=np.int32,
    )

    values = np.asarray(
        [item[1] for item in frame_values],
        dtype=np.float32,
    )

    window = max(
        1,
        int(round(fps * window_seconds)),
    )

    if window <= 1:
        smoothed = values
    else:
        kernel = np.ones(
            window,
            dtype=np.float32,
        ) / float(window)

        smoothed = np.convolve(
            values,
            kernel,
            mode="same",
        )

    return list(
        zip(
            frames.tolist(),
            smoothed.tolist(),
        )
    )


def normalized_sequence(frame_values, fps, baseline_percentile):
    smoothed = smooth_sequence(
        frame_values,
        fps,
        window_seconds=0.20,
    )

    if not smoothed:
        return [], 0.0

    values = np.asarray(
        [value for _, value in smoothed],
        dtype=np.float32,
    )

    baseline = float(
        np.percentile(
            values,
            baseline_percentile,
        )
    )

    if baseline <= 1e-6:
        return [], baseline

    normalized = [
        (
            frame_idx,
            float(value / baseline),
        )
        for frame_idx, value in smoothed
    ]

    return normalized, baseline


def longest_run_seconds(
    frame_values,
    threshold_ratio,
    fps,
):
    if not frame_values or fps <= 0:
        return 0.0

    longest = 0
    current = 0
    previous_frame = None

    for frame_idx, value in frame_values:
        consecutive = (
            previous_frame is not None
            and frame_idx == previous_frame + 1
        )

        if value >= threshold_ratio:
            current = (
                current + 1
                if consecutive
                else 1
            )
            longest = max(
                longest,
                current,
            )
        else:
            current = 0

        previous_frame = frame_idx

    return longest / fps


def predict_rows(
    rows,
    frames_by_video,
    baseline_percentile,
    threshold_ratio,
    min_duration,
):
    predictions = []

    for row in rows:
        sequence, baseline = normalized_sequence(
            frames_by_video.get(
                row["video"],
                [],
            ),
            row["fps"],
            baseline_percentile,
        )

        run_seconds = longest_run_seconds(
            sequence,
            threshold_ratio,
            row["fps"],
        )

        prediction = (
            "yawning"
            if run_seconds >= min_duration
            else "no_yawn"
        )

        max_ratio = (
            max(
                (value for _, value in sequence),
                default=0.0,
            )
        )

        predictions.append(
            {
                **row,
                "baseline_mar": baseline,
                "max_ratio": max_ratio,
                "max_run_seconds": run_seconds,
                "prediction": prediction,
            }
        )

    return predictions


def metrics(rows):
    tp = tn = fp = fn = 0
    fp_normal = 0
    fp_talking = 0

    for row in rows:
        truth = (
            row["binary_label"]
            == "yawning"
        )
        pred = (
            row["prediction"]
            == "yawning"
        )

        if truth and pred:
            tp += 1
        elif (not truth) and (not pred):
            tn += 1
        elif (not truth) and pred:
            fp += 1

            if row["original_label"] == "normal":
                fp_normal += 1
            elif row["original_label"] == "talking":
                fp_talking += 1
        else:
            fn += 1

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else 0.0
    )

    balanced_accuracy = (
        sensitivity + specificity
    ) / 2.0

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * sensitivity
        / (precision + sensitivity)
        if (precision + sensitivity)
        else 0.0
    )

    total = tp + tn + fp + fn
    accuracy = (
        (tp + tn) / total
        if total
        else 0.0
    )

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "fp_normal": fp_normal,
        "fp_talking": fp_talking,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "f1": f1,
    }


def mean(values):
    return (
        sum(values) / len(values)
        if values
        else 0.0
    )


def main():
    if not FRAMES_CSV.exists():
        raise FileNotFoundError(
            FRAMES_CSV
        )

    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            SUMMARY_CSV
        )

    summary_rows = load_summary()
    frames_by_video = load_frames()

    dev_rows = [
        row
        for row in summary_rows
        if row["split"] == "dev"
    ]

    validation_rows = [
        row
        for row in summary_rows
        if row["split"] == "validation"
    ]

    print("=" * 78)
    print("A4 - MAR NORMALIZADO + VALIDACION CRUZADA POR PARTICIPANTE")
    print("=" * 78)
    print(
        f"DEV para CV:        "
        f"{len(dev_rows)} videos / "
        f"{len(DEV_SUBJECTS)} sujetos"
    )
    print(
        f"VALIDACION externa: "
        f"{len(validation_rows)} videos / "
        f"{len(VALIDATION_SUBJECTS)} sujetos"
    )
    print()

    baseline_percentiles = [
        10,
        15,
        20,
        25,
        30,
    ]

    ratio_thresholds = [
        round(
            1.20 + 0.05 * i,
            2,
        )
        for i in range(37)
    ]

    durations = [
        0.40,
        0.60,
        0.80,
        1.00,
        1.20,
        1.40,
        1.60,
        1.80,
        2.00,
        2.25,
        2.50,
        2.75,
        3.00,
        3.50,
        4.00,
    ]

    sweep = []

    for baseline_percentile in baseline_percentiles:
        for ratio_threshold in ratio_thresholds:
            for duration in durations:
                fold_metrics = []

                for held_subject in DEV_SUBJECTS:
                    fold_rows = [
                        row
                        for row in dev_rows
                        if row["subject_id"]
                        == held_subject
                    ]

                    predictions = predict_rows(
                        fold_rows,
                        frames_by_video,
                        baseline_percentile,
                        ratio_threshold,
                        duration,
                    )

                    fold_metrics.append(
                        metrics(
                            predictions
                        )
                    )

                row = {
                    "baseline_percentile": baseline_percentile,
                    "ratio_threshold": ratio_threshold,
                    "min_duration_seconds": duration,
                    "mean_balanced_accuracy": mean(
                        [
                            m[
                                "balanced_accuracy"
                            ]
                            for m in fold_metrics
                        ]
                    ),
                    "min_fold_balanced_accuracy": min(
                        [
                            m[
                                "balanced_accuracy"
                            ]
                            for m in fold_metrics
                        ]
                    ),
                    "mean_f1": mean(
                        [
                            m["f1"]
                            for m in fold_metrics
                        ]
                    ),
                    "mean_sensitivity": mean(
                        [
                            m["sensitivity"]
                            for m in fold_metrics
                        ]
                    ),
                    "mean_specificity": mean(
                        [
                            m["specificity"]
                            for m in fold_metrics
                        ]
                    ),
                }

                sweep.append(row)

    best = max(
        sweep,
        key=lambda row: (
            row[
                "mean_balanced_accuracy"
            ],
            row[
                "min_fold_balanced_accuracy"
            ],
            row["mean_f1"],
            row["mean_sensitivity"],
            row["mean_specificity"],
            -abs(
                row[
                    "baseline_percentile"
                ] - 20
            ),
            -row["ratio_threshold"],
            -row[
                "min_duration_seconds"
            ],
        ),
    )

    SWEEP_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SWEEP_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        fields = [
            "baseline_percentile",
            "ratio_threshold",
            "min_duration_seconds",
            "mean_balanced_accuracy",
            "min_fold_balanced_accuracy",
            "mean_f1",
            "mean_sensitivity",
            "mean_specificity",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(sweep)

    baseline_percentile = int(
        best["baseline_percentile"]
    )
    ratio_threshold = float(
        best["ratio_threshold"]
    )
    duration = float(
        best[
            "min_duration_seconds"
        ]
    )

    dev_predictions = predict_rows(
        dev_rows,
        frames_by_video,
        baseline_percentile,
        ratio_threshold,
        duration,
    )

    validation_predictions = predict_rows(
        validation_rows,
        frames_by_video,
        baseline_percentile,
        ratio_threshold,
        duration,
    )

    dev_metrics = metrics(
        dev_predictions
    )
    validation_metrics = metrics(
        validation_predictions
    )

    with PREDICTIONS_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        fields = [
            "subject_id",
            "original_label",
            "binary_label",
            "prediction",
            "video",
            "baseline_mar",
            "max_ratio",
            "max_run_seconds",
            "correct",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        for row in validation_predictions:
            writer.writerow(
                {
                    "subject_id": row[
                        "subject_id"
                    ],
                    "original_label": row[
                        "original_label"
                    ],
                    "binary_label": row[
                        "binary_label"
                    ],
                    "prediction": row[
                        "prediction"
                    ],
                    "video": row[
                        "video"
                    ],
                    "baseline_mar": round(
                        row[
                            "baseline_mar"
                        ],
                        4,
                    ),
                    "max_ratio": round(
                        row["max_ratio"],
                        3,
                    ),
                    "max_run_seconds": round(
                        row[
                            "max_run_seconds"
                        ],
                        3,
                    ),
                    "correct": (
                        row[
                            "binary_label"
                        ]
                        == row[
                            "prediction"
                        ]
                    ),
                }
            )

    print("CONFIGURACION ELEGIDA SOLO CON DEV-CV")
    print("-" * 78)
    print(
        f"Percentil basal MAR:         "
        f"P{baseline_percentile}"
    )
    print(
        f"Umbral MAR relativo:         "
        f"{ratio_threshold:.2f} x basal"
    )
    print(
        f"Duracion minima:             "
        f"{duration:.2f} s"
    )
    print(
        f"Balanced accuracy CV media:  "
        f"{best['mean_balanced_accuracy'] * 100:.1f}%"
    )
    print(
        f"Peor fold por sujeto:        "
        f"{best['min_fold_balanced_accuracy'] * 100:.1f}%"
    )
    print(
        f"Sensibilidad CV media:       "
        f"{best['mean_sensitivity'] * 100:.1f}%"
    )
    print(
        f"Especificidad CV media:      "
        f"{best['mean_specificity'] * 100:.1f}%"
    )
    print(
        f"F1 CV medio:                 "
        f"{best['mean_f1']:.3f}"
    )

    print()
    print("RENDIMIENTO SOBRE TODO DEV")
    print("-" * 78)
    print(
        f"Accuracy:          "
        f"{dev_metrics['accuracy'] * 100:.1f}%"
    )
    print(
        f"Balanced accuracy: "
        f"{dev_metrics['balanced_accuracy'] * 100:.1f}%"
    )
    print(
        f"Sensibilidad:      "
        f"{dev_metrics['sensitivity'] * 100:.1f}%"
    )
    print(
        f"Especificidad:     "
        f"{dev_metrics['specificity'] * 100:.1f}%"
    )
    print(
        f"Confusion DEV:     "
        f"TP={dev_metrics['tp']} "
        f"TN={dev_metrics['tn']} "
        f"FP={dev_metrics['fp']} "
        f"FN={dev_metrics['fn']}"
    )

    print()
    print("VALIDACION EN LOS 4 SUJETOS APARTADOS")
    print("-" * 78)
    print(
        f"Accuracy:          "
        f"{validation_metrics['accuracy'] * 100:.1f}%"
    )
    print(
        f"Balanced accuracy: "
        f"{validation_metrics['balanced_accuracy'] * 100:.1f}%"
    )
    print(
        f"Sensibilidad:      "
        f"{validation_metrics['sensitivity'] * 100:.1f}%"
    )
    print(
        f"Especificidad:     "
        f"{validation_metrics['specificity'] * 100:.1f}%"
    )
    print(
        f"Precision:         "
        f"{validation_metrics['precision'] * 100:.1f}%"
    )
    print(
        f"F1:                "
        f"{validation_metrics['f1']:.3f}"
    )
    print(
        f"Confusion VALID:   "
        f"TP={validation_metrics['tp']} "
        f"TN={validation_metrics['tn']} "
        f"FP={validation_metrics['fp']} "
        f"FN={validation_metrics['fn']}"
    )
    print(
        f"FP por clase:      "
        f"Normal={validation_metrics['fp_normal']} "
        f"Talking={validation_metrics['fp_talking']}"
    )

    print()
    print("PREDICCIONES VALIDACION")
    print("-" * 78)

    for row in validation_predictions:
        status = (
            "OK"
            if row[
                "binary_label"
            ]
            == row[
                "prediction"
            ]
            else "ERROR"
        )

        print(
            f"{row['subject_id']} | "
            f"{row['original_label']:7s} | "
            f"base={row['baseline_mar']:.3f} | "
            f"maxR={row['max_ratio']:.2f} | "
            f"run={row['max_run_seconds']:.2f}s | "
            f"pred={row['prediction']:7s} | "
            f"{status}"
        )

    print()
    print(
        f"CV sweep:     "
        f"{SWEEP_CSV.relative_to(BASE_DIR)}"
    )
    print(
        f"Predicciones: "
        f"{PREDICTIONS_CSV.relative_to(BASE_DIR)}"
    )


if __name__ == "__main__":
    main()
