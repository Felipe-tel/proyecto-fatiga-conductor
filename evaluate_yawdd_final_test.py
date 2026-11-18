from pathlib import Path
import csv
from collections import defaultdict

import numpy as np

BASE_DIR = Path(__file__).resolve().parent

FRAMES_CSV = BASE_DIR / "data" / "features" / "yawdd_mar_frames.csv"
SUMMARY_CSV = BASE_DIR / "data" / "features" / "yawdd_mar_summary.csv"

OUTPUT_CSV = (
    BASE_DIR
    / "data"
    / "features"
    / "yawdd_final_test_predictions.csv"
)

FINAL_TEST_SUBJECTS = {
    "F11",
    "F12",
    "M17",
    "M18",
}

# Hiperparametros congelados antes de ver estos sujetos.
BASELINE_PERCENTILE = 20
RATIO_THRESHOLD = 1.30
MIN_DURATION_SECONDS = 2.50
SMOOTH_WINDOW_SECONDS = 0.20


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
            subject_id = subject_id_from_filename(row["video"])

            if subject_id not in FINAL_TEST_SUBJECTS:
                continue

            rows.append(
                {
                    "video": row["video"],
                    "original_label": row["label"],
                    "binary_label": binary_label(row["label"]),
                    "subject_id": subject_id,
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
            subject_id = subject_id_from_filename(row["video"])

            if subject_id not in FINAL_TEST_SUBJECTS:
                continue

            by_video[row["video"]].append(
                (
                    int(row["frame"]),
                    float(row["mar"]),
                )
            )

    for video in by_video:
        by_video[video].sort(key=lambda x: x[0])

    return dict(by_video)


def normalized_sequence(frame_values, fps):
    if not frame_values:
        return [], 0.0

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
        int(round(fps * SMOOTH_WINDOW_SECONDS)),
    )

    if window > 1:
        kernel = np.ones(
            window,
            dtype=np.float32,
        ) / float(window)

        values = np.convolve(
            values,
            kernel,
            mode="same",
        )

    baseline = float(
        np.percentile(
            values,
            BASELINE_PERCENTILE,
        )
    )

    if baseline <= 1e-6:
        return [], baseline

    sequence = [
        (
            int(frame_idx),
            float(value / baseline),
        )
        for frame_idx, value in zip(frames, values)
    ]

    return sequence, baseline


def longest_run_seconds(sequence, fps):
    longest = 0
    current = 0
    previous_frame = None

    for frame_idx, ratio in sequence:
        consecutive = (
            previous_frame is not None
            and frame_idx == previous_frame + 1
        )

        if ratio >= RATIO_THRESHOLD:
            current = current + 1 if consecutive else 1
            longest = max(longest, current)
        else:
            current = 0

        previous_frame = frame_idx

    return longest / fps if fps > 0 else 0.0


def metrics(rows):
    tp = tn = fp = fn = 0
    fp_normal = 0
    fp_talking = 0

    for row in rows:
        truth = row["binary_label"] == "yawning"
        pred = row["prediction"] == "yawning"

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

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (sensitivity + specificity) / 2.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    f1 = (
        2.0 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity)
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


def main():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            "Falta yawdd_mar_summary.csv. "
            "Ejecuta primero extract_yawdd_mar.py."
        )

    if not FRAMES_CSV.exists():
        raise FileNotFoundError(
            "Falta yawdd_mar_frames.csv. "
            "Ejecuta primero extract_yawdd_mar.py."
        )

    rows = load_summary()
    frames_by_video = load_frames()

    if len(rows) != 12:
        raise RuntimeError(
            f"Se esperaban 12 videos de test final y se encontraron {len(rows)}. "
            "Regenera manifest y MAR despues de descargar los sujetos nuevos."
        )

    predictions = []

    for row in rows:
        sequence, baseline = normalized_sequence(
            frames_by_video.get(row["video"], []),
            row["fps"],
        )

        max_ratio = max(
            (ratio for _, ratio in sequence),
            default=0.0,
        )

        run_seconds = longest_run_seconds(
            sequence,
            row["fps"],
        )

        prediction = (
            "yawning"
            if run_seconds >= MIN_DURATION_SECONDS
            else "no_yawn"
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

    result = metrics(predictions)

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_CSV.open(
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

        for row in predictions:
            writer.writerow(
                {
                    "subject_id": row["subject_id"],
                    "original_label": row["original_label"],
                    "binary_label": row["binary_label"],
                    "prediction": row["prediction"],
                    "video": row["video"],
                    "baseline_mar": round(row["baseline_mar"], 4),
                    "max_ratio": round(row["max_ratio"], 3),
                    "max_run_seconds": round(row["max_run_seconds"], 3),
                    "correct": row["binary_label"] == row["prediction"],
                }
            )

    print("=" * 78)
    print("A4 - TEST FINAL YAWDD CON SUJETOS NUNCA VISTOS")
    print("=" * 78)
    print("Hiperparametros CONGELADOS antes del test:")
    print(f"  Percentil basal:      P{BASELINE_PERCENTILE}")
    print(f"  MAR relativo:         {RATIO_THRESHOLD:.2f} x basal")
    print(f"  Duracion minima:      {MIN_DURATION_SECONDS:.2f} s")
    print(f"  Suavizado:            {SMOOTH_WINDOW_SECONDS:.2f} s")
    print()

    print("RESULTADOS")
    print("-" * 78)
    print(f"Accuracy:          {result['accuracy'] * 100:.1f}%")
    print(f"Balanced accuracy: {result['balanced_accuracy'] * 100:.1f}%")
    print(f"Sensibilidad:      {result['sensitivity'] * 100:.1f}%")
    print(f"Especificidad:     {result['specificity'] * 100:.1f}%")
    print(f"Precision:         {result['precision'] * 100:.1f}%")
    print(f"F1:                {result['f1']:.3f}")
    print(
        f"Confusion FINAL:   TP={result['tp']} TN={result['tn']} "
        f"FP={result['fp']} FN={result['fn']}"
    )
    print(
        f"FP por clase:      Normal={result['fp_normal']} "
        f"Talking={result['fp_talking']}"
    )

    print()
    print("PREDICCIONES")
    print("-" * 78)

    for row in predictions:
        status = (
            "OK"
            if row["binary_label"] == row["prediction"]
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
    print(f"Predicciones: {OUTPUT_CSV.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
