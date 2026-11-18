from pathlib import Path
import csv
import math

import cv2
import mediapipe as mp
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (BASE_DIR / "models" / "face_landmarker.task").resolve()

RAW_ROOT = BASE_DIR / "data" / "raw" / "yawdd"
FEATURES_DIR = BASE_DIR / "data" / "features"

FRAMES_CSV = FEATURES_DIR / "yawdd_mar_frames.csv"
SUMMARY_CSV = FEATURES_DIR / "yawdd_mar_summary.csv"

MOUTH_MAR_INDICES = [61, 291, 37, 84, 0, 17, 267, 314]


def distancia(p1, p2):
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def calcular_mar(landmarks, width, height):
    puntos = {}

    for idx in MOUTH_MAR_INDICES:
        if idx >= len(landmarks):
            return None

        landmark = landmarks[idx]
        puntos[idx] = (
            float(landmark.x * width),
            float(landmark.y * height),
        )

    horizontal = distancia(puntos[61], puntos[291])

    if horizontal <= 0:
        return None

    v1 = distancia(puntos[37], puntos[84])
    v2 = distancia(puntos[0], puntos[17])
    v3 = distancia(puntos[267], puntos[314])

    return float((v1 + v2 + v3) / (3.0 * horizontal))


def percentil(valores, q):
    if not valores:
        return None

    return float(np.percentile(np.asarray(valores, dtype=np.float32), q))


def resumir_video(video_path, label, landmarker):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir: {video_path.name}")
        return None, []

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 0
    detected_frames = 0
    mar_values = []
    frame_rows = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        timestamp_ms = int(frame_idx * 1000.0 / fps)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        mar = None

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            mar = calcular_mar(landmarks, width, height)

        if mar is not None:
            detected_frames += 1
            mar_values.append(mar)

            frame_rows.append(
                {
                    "label": label,
                    "video": video_path.name,
                    "frame": frame_idx,
                    "time_seconds": round(frame_idx / fps, 4),
                    "mar": round(mar, 6),
                }
            )

        frame_idx += 1

    cap.release()

    processed_frames = frame_idx

    detection_rate = (
        detected_frames / processed_frames * 100.0
        if processed_frames > 0
        else 0.0
    )

    summary = {
        "label": label,
        "video": video_path.name,
        "frames_reported": total_frames,
        "frames_processed": processed_frames,
        "frames_with_face": detected_frames,
        "face_detection_rate_pct": round(detection_rate, 2),
        "fps": round(fps, 2),
        "mar_mean": round(float(np.mean(mar_values)), 6) if mar_values else "",
        "mar_median": round(float(np.median(mar_values)), 6) if mar_values else "",
        "mar_p90": round(percentil(mar_values, 90), 6) if mar_values else "",
        "mar_p95": round(percentil(mar_values, 95), 6) if mar_values else "",
        "mar_max": round(max(mar_values), 6) if mar_values else "",
        "pct_mar_ge_020": round(
            sum(value >= 0.20 for value in mar_values) / len(mar_values) * 100.0, 2
        ) if mar_values else "",
        "pct_mar_ge_025": round(
            sum(value >= 0.25 for value in mar_values) / len(mar_values) * 100.0, 2
        ) if mar_values else "",
        "pct_mar_ge_030": round(
            sum(value >= 0.30 for value in mar_values) / len(mar_values) * 100.0, 2
        ) if mar_values else "",
        "pct_mar_ge_035": round(
            sum(value >= 0.35 for value in mar_values) / len(mar_values) * 100.0, 2
        ) if mar_values else "",
        "pct_mar_ge_040": round(
            sum(value >= 0.40 for value in mar_values) / len(mar_values) * 100.0, 2
        ) if mar_values else "",
        "pct_mar_ge_050": round(
            sum(value >= 0.50 for value in mar_values) / len(mar_values) * 100.0, 2
        ) if mar_values else "",
        "pct_mar_ge_060": round(
            sum(value >= 0.60 for value in mar_values) / len(mar_values) * 100.0, 2
        ) if mar_values else "",
    }

    return summary, frame_rows


def crear_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return FaceLandmarker.create_from_options(options)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe el modelo: {MODEL_PATH}")

    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"No existe YawDD: {RAW_ROOT}")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    videos = []

    for label in ("normal", "talking", "yawning"):
        folder = RAW_ROOT / label

        if not folder.exists():
            continue

        for path in sorted(folder.glob("*.avi")):
            videos.append((label, path))

    print("=" * 66)
    print("EXTRACCION MAR - YAWDD")
    print("=" * 66)
    print(f"Videos encontrados: {len(videos)}")
    print()

    all_summaries = []
    all_frames = []

    for index, (label, video_path) in enumerate(videos, start=1):
        print(
            f"[{index:02d}/{len(videos):02d}] "
            f"{label:7s} | {video_path.name}"
        )

        with crear_landmarker() as landmarker:
            summary, frame_rows = resumir_video(
                video_path,
                label,
                landmarker,
            )

        if summary is not None:
            all_summaries.append(summary)
            all_frames.extend(frame_rows)

            print(
                "    rostro: "
                f"{summary['face_detection_rate_pct']}%"
                " | MAR mediana: "
                f"{summary['mar_median']}"
                " | P95: "
                f"{summary['mar_p95']}"
                " | max: "
                f"{summary['mar_max']}"
            )

    summary_fields = [
        "label",
        "video",
        "frames_reported",
        "frames_processed",
        "frames_with_face",
        "face_detection_rate_pct",
        "fps",
        "mar_mean",
        "mar_median",
        "mar_p90",
        "mar_p95",
        "mar_max",
        "pct_mar_ge_020",
        "pct_mar_ge_025",
        "pct_mar_ge_030",
        "pct_mar_ge_035",
        "pct_mar_ge_040",
        "pct_mar_ge_050",
        "pct_mar_ge_060",
    ]

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(all_summaries)

    frame_fields = [
        "label",
        "video",
        "frame",
        "time_seconds",
        "mar",
    ]

    with FRAMES_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=frame_fields)
        writer.writeheader()
        writer.writerows(all_frames)

    counts = {
        label: sum(1 for row in all_summaries if row["label"] == label)
        for label in ("normal", "talking", "yawning")
    }

    print()
    print("=" * 66)
    print("RESUMEN")
    print("=" * 66)
    print(f"Normal procesados:  {counts['normal']}")
    print(f"Talking procesados: {counts['talking']}")
    print(f"Yawning procesados: {counts['yawning']}")
    print(f"Frames MAR validos: {len(all_frames)}")
    print()
    print(f"Resumen: {SUMMARY_CSV.relative_to(BASE_DIR)}")
    print(f"Frames:  {FRAMES_CSV.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
