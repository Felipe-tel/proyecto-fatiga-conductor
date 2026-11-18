"""Extracción de landmarks faciales con MediaPipe Face Landmarker (API Tasks).

Las versiones recientes de MediaPipe ya no incluyen `mp.solutions.face_mesh`;
se usa `FaceLandmarker`, que entrega 478 puntos (468 de la malla + 10 de iris),
opcionalmente 52 blendshapes (por ejemplo `jawOpen`) y la matriz de
transformación facial 4x4, de la que se obtiene la pose de la cabeza.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


@dataclass
class FaceFrame:
    pts: np.ndarray          # (478, 2) en píxeles
    blendshapes: dict        # nombre -> score [0, 1]
    pose: tuple = None       # (yaw, pitch, roll) en grados, o None


def landmarks_to_array(face_landmarks, w, h):
    """Convierte la lista de NormalizedLandmark de MediaPipe a un arreglo (N, 2) en píxeles."""
    return np.array([[p.x * w, p.y * h] for p in face_landmarks], dtype=np.float32)


def blendshapes_to_dict(result):
    if not getattr(result, "face_blendshapes", None):
        return {}
    return {c.category_name: c.score for c in result.face_blendshapes[0]}


def pose_from_matrix(m):
    """(yaw, pitch, roll) en grados a partir de la matriz de transformación facial 4x4.

    Solo se usa la magnitud para descartar cuadros con la cabeza muy girada,
    así que el signo de cada ángulo no importa para A4.
    """
    r = np.asarray(m, dtype=float)[:3, :3]
    sy = np.hypot(r[0, 0], r[1, 0])
    pitch = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    yaw = np.degrees(np.arctan2(-r[2, 0], sy))
    roll = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    return float(yaw), float(pitch), float(roll)


def pose_from_result(result):
    mats = getattr(result, "facial_transformation_matrixes", None)
    return pose_from_matrix(mats[0]) if mats else None


class FaceTracker:
    """Envuelve FaceLandmarker en modo VIDEO (usa seguimiento entre cuadros)."""

    def __init__(self, model_path=MODEL_PATH, video=True):
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.lm = vision.FaceLandmarker.create_from_options(opts)
        self.video = video
        self._last_ms = -1

    def process(self, frame_bgr, t_ms=0):
        h, w = frame_bgr.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if self.video:
            # el modo VIDEO exige marcas de tiempo estrictamente crecientes
            t_ms = max(int(t_ms), self._last_ms + 1)
            self._last_ms = t_ms
            res = self.lm.detect_for_video(img, t_ms)
        else:
            res = self.lm.detect(img)
        if not res.face_landmarks:
            return None
        return FaceFrame(landmarks_to_array(res.face_landmarks[0], w, h), blendshapes_to_dict(res),
                         pose_from_result(res))

    def close(self):
        self.lm.close()
