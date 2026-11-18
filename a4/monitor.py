"""Monitor A4 cuadro a cuadro: bostezos + mirada, alertas e indicadores para A6.

Uso mínimo (dentro de cualquier bucle de video que ya tenga los landmarks):

    mon = A4Monitor()
    estado = mon.update(pts, t_s, pose=pose)   # pts: (478, 2) en píxeles o None
    mon.draw(frame, estado, pts)
    ...
    resumen = mon.finish()
    mon.save("resultados/")

Los tiempos pueden ser absolutos (time.time()) o relativos: internamente todo
se mide desde el primer cuadro.

Indicadores para A6: cada `window_s` segundos se cierra una ventana y se
agrega una fila a `mon.rows` (ver INDICADORES). Si ya hay línea base de la
mirada, se agregan los cocientes `*_rel` (valor / línea base del mismo
conductor), que son los que conviene usar para clasificar.
"""
import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .boca import MOUTH_PTS, YawnDetector, mar
from .mirada import IRIS_CENTERS, GazeAnalyzer

# columnas del vector de indicadores por ventana (en este orden)
INDICADORES = [
    "t_ini", "t_fin", "validos_pct",
    "mar_medio", "mar_p95", "jaw_open_p95", "bostezo_activo", "bostezos_ventana", "bostezos_15min",
    "fijacion_media_s", "fijacion_max_s", "fijacion_actual_s", "fijaciones", "sacadas_por_s",
    "vel_pico_media", "amp_sacada_media_deg", "dispersion_h_deg", "dispersion_v_deg", "indice_nistagmo",
    "fijacion_media_rel", "sacadas_rel", "dispersion_h_rel", "indice_nistagmo_rel",
    "linea_base", "nivel_alerta",
]
REL = {"fijacion_media_rel": "fijacion_media_s", "sacadas_rel": "sacadas_por_s",
       "dispersion_h_rel": "dispersion_h_deg", "indice_nistagmo_rel": "indice_nistagmo"}


def _nan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def _clean(o):
    """NaN -> None y tipos de NumPy -> nativos, para escribir JSON válido."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if math.isnan(o) else float(o)
    return o


@dataclass
class EstadoA4:
    t: float
    rostro: bool
    mar: float = float("nan")
    mar_suav: float = float("nan")
    bostezando: bool = False
    boca_abierta_s: float = 0.0
    mirada_h: float = float("nan")
    mirada_v: float = float("nan")
    ear: float = float("nan")
    calibrando_s: float = 0.0        # segundos de calibración de la boca que faltan
    linea_base_s: float = 0.0        # segundos que faltan para la línea base de la mirada
    alertas: list = field(default_factory=list)
    nivel: int = 0                   # 0 normal, 1 aviso, 2 alerta
    indicadores: dict = None         # último vector calculado (ventana deslizante)


class A4Monitor:
    """Junta el detector de bostezos y el análisis de mirada.

    calib_s: segundos iniciales con la boca en reposo para personalizar el umbral de MAR.
    window_s: largo de la ventana de indicadores (tumbling para A6, deslizante en pantalla).
    refresh_s: cada cuánto se recalculan los indicadores en vivo.
    baseline_s: segundos (después de la calibración) que forman la línea base de la mirada.
    yawn_window_s / yawn_alert_n: alerta de fatiga si hay >= n bostezos en esa ventana. Por defecto,
        3 bostezos en 15 min: la definición clínica de frecuencia anormal de bostezos
        (Krestel, Bassetti y Walusinski; ver docs/A4.md).
    stare_s: alerta de mirada fija si una fijación dura más que esto.
    history_s: memoria máxima de la serie de mirada (None = todo; usar ~120 en vivo).
    """

    def __init__(self, fps=30.0, calib_s=5.0, window_s=10.0, refresh_s=1.0, baseline_s=30.0,
                 yawn_window_s=900.0, yawn_alert_n=3, stare_s=6.0, history_s=None,
                 vel_thr=30.0, min_amp_deg=4.0, yawn_kwargs=None, keep_frames=False):
        self.yawn = YawnDetector(fps=fps, calib_s=calib_s, **(yawn_kwargs or {}))
        self.gaze = GazeAnalyzer(fps, history_s=history_s)
        self.calib_s, self.window_s, self.refresh_s, self.baseline_s = calib_s, window_s, refresh_s, baseline_s
        self.yawn_window_s, self.yawn_alert_n, self.stare_s = yawn_window_s, yawn_alert_n, stare_s
        self.vel_thr, self.min_amp_deg = vel_thr, min_amp_deg
        self.keep_frames = keep_frames
        self.frames, self.rows = [], []
        self.baseline = None
        self._t0 = None
        self._t_last = 0.0
        self._next_refresh = 0.0
        self._next_window = window_s
        self._mar_buf = deque()
        self._live = None

    # ------------------------------------------------------------------ cuadro a cuadro
    def update(self, pts, t_s, pose=None, blendshapes=None):
        if self._t0 is None:
            self._t0 = t_s
        t = t_s - self._t0
        self._t_last = t
        st = EstadoA4(t=t, rostro=pts is not None)
        if pts is None:
            self.gaze.update(None, t)
        else:
            st.mar = mar(pts)
            st.mar_suav, st.bostezando = self.yawn.update(st.mar, t)
            st.boca_abierta_s = self.yawn.open_for(t)
            st.mirada_h, st.mirada_v, st.ear = self.gaze.update(pts, t, pose)
            jaw = (blendshapes or {}).get("jawOpen", float("nan"))
            self._mar_buf.append((t, st.mar, jaw))
        while self._mar_buf and self._mar_buf[0][0] < t - max(self.window_s, 60.0):
            self._mar_buf.popleft()
        st.calibrando_s = self.yawn.calib_left(t)

        # cierre de ventanas fijas (las que recibe A6)
        while t >= self._next_window:
            self.rows.append(self.indicadores(self._next_window - self.window_s, self._next_window))
            self._next_window += self.window_s
        # línea base de la mirada y refresco de los indicadores en vivo
        if t >= self._next_refresh:
            self._next_refresh = t + self.refresh_s
            self._try_baseline(t)
            self._live = self.indicadores(max(0.0, t - self.window_s), t + 1e-6)
        st.linea_base_s = 0.0 if self.baseline else max(0.0, self.calib_s + self.baseline_s - t)
        st.indicadores = self._live
        st.alertas, st.nivel = self._alertas(st)
        if self.keep_frames:
            self.frames.append({"t": t, "rostro": int(st.rostro), "mar": st.mar, "mar_suav": st.mar_suav,
                                "bostezo": int(st.bostezando), "mirada_h": st.mirada_h, "mirada_v": st.mirada_v,
                                "ear": st.ear, "yaw": pose[0] if pose else np.nan,
                                "pitch": pose[1] if pose else np.nan, "nivel": st.nivel})
        return st

    def _try_baseline(self, t):
        if self.baseline is not None or t < self.calib_s + self.baseline_s:
            return
        # promedio de ventanas del mismo largo que las de los indicadores: las fijaciones se
        # cortan en el borde de la ventana, así que comparar largos distintos sesga los cocientes
        n = max(1, int(round(self.baseline_s / self.window_s)))
        t_ini = t - n * self.window_s
        partes = [self.gaze.metrics(t_ini + k * self.window_s, t_ini + (k + 1) * self.window_s,
                                    self.vel_thr, self.min_amp_deg) for k in range(n)]
        partes = [g for g in partes if g["validos_pct"] >= 50]
        if len(partes) * 2 >= n:  # si no hay datos suficientes, se reintenta en el siguiente refresco
            with np.errstate(all="ignore"):
                self.baseline = {k: float(np.nanmean([g[k] for g in partes])) if any(not _nan(g[k]) for g in partes)
                                 else np.nan for k in partes[0]}
            self.baseline.update(t_ini=t_ini, t_fin=t, ventanas=len(partes))

    # ------------------------------------------------------------------ indicadores
    def indicadores(self, t0, t1):
        g = self.gaze.metrics(t0, t1, self.vel_thr, self.min_amp_deg)
        mars = np.array([m for tt, m, _ in self._mar_buf if t0 <= tt < t1])
        jaws = np.array([j for tt, _, j in self._mar_buf if t0 <= tt < t1 and not math.isnan(j)])
        d = {"t_ini": float(t0), "t_fin": float(t1),
             "mar_medio": float(mars.mean()) if len(mars) else np.nan,
             "mar_p95": float(np.percentile(mars, 95)) if len(mars) else np.nan,
             "jaw_open_p95": float(np.percentile(jaws, 95)) if len(jaws) else np.nan,
             "bostezo_activo": int(self.yawn.yawning(t1)),
             "bostezos_ventana": self.yawn.count(t0, t1),
             "bostezos_15min": self.yawn.count(t1 - self.yawn_window_s, t1),
             **g}
        for k, base_k in REL.items():
            b = self.baseline.get(base_k) if self.baseline else None
            d[k] = float(d[base_k] / b) if not _nan(b) and b > 0 and not _nan(d[base_k]) else np.nan
        d["linea_base"] = int(self.baseline is not None)
        d["nivel_alerta"] = self._nivel_ventana(d)
        return {k: d.get(k, np.nan) for k in INDICADORES}

    def _nivel_ventana(self, d):
        nivel = 0
        if d["bostezos_ventana"] > 0 or d["fijacion_max_s"] >= self.stare_s:
            nivel = 1
        if d["bostezos_15min"] >= self.yawn_alert_n or self._mirada_anomala(d):
            nivel = 2
        return nivel

    def _mirada_anomala(self, d):
        """Exploración visual reducida respecto de la línea base (fijaciones 2x más largas
        y la mitad de sacadas). Requiere línea base y al menos 60 % de cuadros válidos."""
        if not d.get("linea_base") or _nan(d["validos_pct"]) or d["validos_pct"] < 60:
            return False
        f, s = d["fijacion_media_rel"], d["sacadas_rel"]
        return not _nan(f) and not _nan(s) and f >= 2.0 and s <= 0.5

    def _oscilacion(self, d):
        if not d.get("linea_base") or _nan(d["indice_nistagmo"]) or d["validos_pct"] < 60:
            return False
        b = self.baseline["indice_nistagmo"]
        return d["indice_nistagmo"] >= max(0.4, (0 if _nan(b) else b) + 0.2)

    def _alertas(self, st):
        al, nivel = [], 0
        if st.bostezando:
            al.append("BOSTEZO"); nivel = 1
        d = st.indicadores
        if d:
            if d["bostezos_15min"] >= self.yawn_alert_n:
                al.append(f"FATIGA: {d['bostezos_15min']} bostezos en {self.yawn_window_s / 60:.0f} min"); nivel = 2
            if d["fijacion_actual_s"] >= self.stare_s:
                al.append(f"MIRADA FIJA {d['fijacion_actual_s']:.1f} s"); nivel = max(nivel, 1)
            if self._mirada_anomala(d):
                al.append("EXPLORACION VISUAL REDUCIDA"); nivel = 2
            if self._oscilacion(d):
                al.append("OSCILACION OCULAR"); nivel = max(nivel, 1)
        return al, nivel

    # ------------------------------------------------------------------ dibujo
    def draw(self, frame, st, pts=None, origin=(15, 150)):
        """Dibuja los puntos de boca e iris y un panel de texto con el estado A4."""
        if pts is not None:
            for idx in MOUTH_PTS:
                cv2.circle(frame, (int(pts[idx][0]), int(pts[idx][1])), 2, (0, 200, 255), -1)
            for idx in IRIS_CENTERS:
                cv2.circle(frame, (int(pts[idx][0]), int(pts[idx][1])), 3, (0, 255, 0), -1)
        font, white, gray = cv2.FONT_HERSHEY_SIMPLEX, (255, 255, 255), (200, 200, 200)
        color = [(0, 255, 0), (0, 165, 255), (0, 0, 255)][st.nivel]
        estado = " | ".join(st.alertas) if st.alertas else "NORMAL"
        if st.calibrando_s > 0:
            estado = f"calibrando, boca cerrada {st.calibrando_s:.0f} s"
            color = (255, 255, 0)
        d = st.indicadores or {}
        fmt = lambda v, f="{:.2f}": "--" if _nan(v) else f.format(v)
        base = "lista" if self.baseline else f"faltan {st.linea_base_s:.0f} s"
        lineas = [  # (texto, escala, color, grosor)
            (f"A4: {estado}", 0.6, color, 2),
            (f"MAR {fmt(st.mar_suav)} (umbral {self.yawn.thr_on:.2f}) | Bostezos 15 min: "
             f"{d.get('bostezos_15min', 0)}", 0.5, white, 1),
            (f"Fijacion {fmt(d.get('fijacion_actual_s'), '{:.1f}')} s | Sacadas {fmt(d.get('sacadas_por_s'))}/s"
             f" | Nistagmo {fmt(d.get('indice_nistagmo'))}", 0.5, white, 1),
            (f"Linea base mirada: {base} | Validos {fmt(d.get('validos_pct'), '{:.0f}')}%", 0.45, gray, 1),
        ]
        # fondo semitransparente para que el texto se lea sobre la cara
        x, y = origin
        ancho = max(cv2.getTextSize(txt, font, s, g)[0][0] for txt, s, _, g in lineas)
        x2, y2 = min(x + ancho + 8, frame.shape[1] - 1), min(y + 24 * (len(lineas) - 1) + 8, frame.shape[0] - 1)
        x1, y1 = max(x - 6, 0), max(y - 20, 0)
        if x2 > x1 and y2 > y1:
            frame[y1:y2, x1:x2] = (frame[y1:y2, x1:x2] * 0.45).astype(frame.dtype)
        for k, (txt, s, c, g) in enumerate(lineas):
            cv2.putText(frame, txt, (x, y + 24 * k), font, s, c, g, cv2.LINE_AA)

    # ------------------------------------------------------------------ cierre
    def finish(self):
        t = self._t_last
        # última ventana incompleta: se agrega si cubre al menos la mitad
        if t - (self._next_window - self.window_s) >= self.window_s / 2:
            self.rows.append(self.indicadores(self._next_window - self.window_s, t + 1e-6))
        events = self.yawn.finish(t)
        return {
            "duracion_s": t,
            "umbral_mar": self.yawn.thr_on,
            "umbral_mar_salida": self.yawn.thr_off,
            "bostezos": [{"inicio_s": e.start_s, "fin_s": e.end_s, "duracion_s": e.duration_s,
                          "mar_pico": e.peak_mar} for e in events],
            "mirada_global": self.gaze.metrics(vel_thr=self.vel_thr, min_amp_deg=self.min_amp_deg),
            "linea_base_mirada": self.baseline,
            "ventanas": len(self.rows),
            "ventanas_con_alerta": sum(1 for r in self.rows if r["nivel_alerta"] > 0),
        }

    def save(self, out_dir, resumen=None):
        import pandas as pd
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows, columns=INDICADORES).to_csv(out / "indicadores_a4.csv", index=False)
        if self.frames:
            pd.DataFrame(self.frames).to_csv(out / "frames_a4.csv", index=False)
        if resumen is not None:
            (out / "resumen_a4.json").write_text(
                json.dumps(_clean(resumen), indent=2, ensure_ascii=False), encoding="utf-8")
        return out
