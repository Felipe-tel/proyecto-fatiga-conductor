"""Análisis de la región bucal: MAR y detección de bostezos por eventos.

MAR (Mouth Aspect Ratio) con labios internos de la malla de MediaPipe:
    MAR = (|81-178| + |13-14| + |311-402|) / (3 * |78-308|)
Con la boca cerrada vale ~0; al hablar sube por instantes (en YawDD el p95 al
hablar llega a ~0.39); en un bostezo supera ~0.4 durante uno o más segundos.
Por eso la decisión combina umbral + duración mínima + histéresis. Los valores
por defecto (0.40 / 0.32 / 1.0 s) son los que eligió la validación cruzada por
sujeto en 42 videos de YawDD (a4_eval_yawdd.py).
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# pares verticales (labio superior interno, labio inferior interno) y comisuras
MOUTH_V = [(81, 178), (13, 14), (311, 402)]
MOUTH_H = (78, 308)
MOUTH_PTS = (78, 308, 13, 14, 81, 178, 311, 402)


def mar(pts):
    v = sum(np.linalg.norm(pts[a] - pts[b]) for a, b in MOUTH_V)
    h = np.linalg.norm(pts[MOUTH_H[0]] - pts[MOUTH_H[1]])
    return float(v / (3.0 * h)) if h > 1e-6 else 0.0


@dataclass
class YawnEvent:
    start_s: float
    end_s: float
    peak_mar: float

    @property
    def duration_s(self):
        return self.end_s - self.start_s


@dataclass
class YawnDetector:
    """Máquina de estados con histéresis sobre el MAR suavizado.

    thr_on / thr_off: umbrales de entrada y salida (histéresis).
    min_dur_s: duración mínima para aceptar un bostezo (descarta habla y canto).
    calib_s: si > 0, los primeros segundos (desde el primer cuadro) fijan un
             umbral personalizado thr_on = mediana + delta, acotado entre el
             umbral fijo (piso) y thr_max (techo). Sirve para personas con la
             boca entreabierta en reposo. Se usa la mediana y no media + k·DE
             porque la mediana no se altera si la persona habla durante la
             calibración.
    Los tiempos pueden ser absolutos (p. ej. time.time()): la calibración se
    mide desde el primer cuadro recibido.
    """
    fps: float = 30.0
    thr_on: float = 0.40
    thr_off: float = 0.32
    min_dur_s: float = 1.0
    smooth_n: int = 5
    calib_s: float = 0.0
    delta: float = 0.30
    thr_max: float = 0.70
    events: list = field(default_factory=list)

    def __post_init__(self):
        self._buf = deque(maxlen=self.smooth_n)
        self._calib = []
        self._t_first = None
        self._open_since = None
        self._peak = 0.0
        self.calibrated = self.calib_s <= 0

    def calib_left(self, t_s):
        if self.calibrated or self._t_first is None:
            return 0.0 if self.calibrated else self.calib_s
        return max(0.0, self.calib_s - (t_s - self._t_first))

    def update(self, value, t_s):
        """Agrega el MAR del cuadro en t_s. Devuelve (mar_suavizado, en_bostezo)."""
        if self._t_first is None:
            self._t_first = t_s
        self._buf.append(value)
        m = float(np.median(self._buf))
        if not self.calibrated:
            if t_s - self._t_first < self.calib_s:
                self._calib.append(m)
                return m, False
            if self._calib:
                base = float(np.median(self._calib))
                self.thr_on = min(max(self.thr_on, base + self.delta), max(self.thr_on, self.thr_max))
                self.thr_off = min(self.thr_off, self.thr_on * 0.8)
            self._calib = []
            self.calibrated = True
        if self._open_since is None:
            if m >= self.thr_on:
                self._open_since, self._peak = t_s, m
        else:
            self._peak = max(self._peak, m)
            if m < self.thr_off:
                self._close(t_s)
        return m, self.yawning(t_s)

    def yawning(self, t_s):
        return self._open_since is not None and (t_s - self._open_since) >= self.min_dur_s

    def open_for(self, t_s):
        """Segundos que lleva la boca abierta sobre el umbral (0 si está cerrada)."""
        return 0.0 if self._open_since is None else t_s - self._open_since

    def count(self, t0, t1):
        """Bostezos terminados que empezaron en [t0, t1), más el que esté en curso."""
        n = sum(1 for e in self.events if t0 <= e.start_s < t1)
        if self._open_since is not None and self.yawning(t1) and self._open_since >= t0:
            n += 1
        return n

    def _close(self, t_s):
        if t_s - self._open_since >= self.min_dur_s:
            self.events.append(YawnEvent(self._open_since, t_s, self._peak))
        self._open_since = None

    def finish(self, t_s):
        if self._open_since is not None:
            self._close(t_s)
        return self.events
