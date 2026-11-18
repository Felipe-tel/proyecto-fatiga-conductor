from pathlib import Path
import csv
import cv2

RAW_ROOT = Path("data/raw/yawdd")
OUTPUT = Path("data/manifests/yawdd_manifest.csv")


def extract_metadata(video_path: Path, label: str):
    cap = cv2.VideoCapture(str(video_path))
    opened = cap.isOpened()

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0

    cap.release()

    duration = frames / fps if fps > 0 else 0.0
    size_mb = video_path.stat().st_size / (1024 * 1024)

    filename = video_path.name
    subject_number = filename.split("-")[0]

    if "Female" in filename:
        sex = "female"
        subject_id = f"F{int(subject_number):02d}"
    elif "Male" in filename:
        sex = "male"
        subject_id = f"M{int(subject_number):02d}"
    else:
        sex = "unknown"
        subject_id = subject_number

    if "NoGlasses" in filename:
        glasses = "no"
    elif "SunGlasses" in filename:
        glasses = "sunglasses"
    elif "Glasses" in filename:
        glasses = "yes"
    else:
        glasses = "unknown"

    valid = (
        opened
        and frames > 0
        and fps > 0
        and width > 0
        and height > 0
    )

    return {
        "subject_id": subject_id,
        "sex": sex,
        "glasses": glasses,
        "label": label,
        "video": filename,
        "frames": frames,
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "duration_seconds": round(duration, 2),
        "size_mb": round(size_mb, 2),
        "valid": valid,
    }


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for label in ("normal", "talking", "yawning"):
        folder = RAW_ROOT / label

        if not folder.exists():
            continue

        for video in sorted(folder.glob("*.avi")):
            rows.append(extract_metadata(video, label))

    rows.sort(key=lambda row: (row["subject_id"], row["label"]))

    fields = [
        "subject_id",
        "sex",
        "glasses",
        "label",
        "video",
        "frames",
        "fps",
        "width",
        "height",
        "duration_seconds",
        "size_mb",
        "valid",
    ]

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    total_videos = len(rows)
    counts = {
        label: sum(1 for row in rows if row["label"] == label)
        for label in ("normal", "talking", "yawning")
    }

    total_frames = sum(row["frames"] for row in rows)
    total_duration = sum(row["duration_seconds"] for row in rows)

    participants = sorted({row["subject_id"] for row in rows})
    females = len({row["subject_id"] for row in rows if row["sex"] == "female"})
    males = len({row["subject_id"] for row in rows if row["sex"] == "male"})
    all_valid = all(row["valid"] for row in rows)

    print("=" * 60)
    print("MANIFEST YAWDD")
    print("=" * 60)
    print(f"Videos totales:       {total_videos}")
    print(f"Normal:               {counts['normal']}")
    print(f"Talking:              {counts['talking']}")
    print(f"Yawning:              {counts['yawning']}")
    print(f"Participantes:        {len(participants)}")
    print(f"Participantes mujer:  {females}")
    print(f"Participantes hombre: {males}")
    print(f"Frames totales:       {total_frames}")
    print(f"Duracion total:       {total_duration / 60:.2f} min")
    print(f"Todos validos:        {all_valid}")
    print()
    print(f"Manifest creado: {OUTPUT}")


if __name__ == "__main__":
    main()
