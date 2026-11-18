"""Seguimiento ocular continuo con los landmarks de iris de MediaPipe.

Posición de la mirada por ojo = posición del centro del iris normalizada
respecto de las comisuras (horizontal) y los párpados (vertical). Al ser
relativa a puntos del mismo ojo, cancela en gran parte la traslación de la
cabeza. La rotación de la cabeza no se cancela: los cuadros con la cabeza muy
girada (según la matriz de transformación de FaceLandmarker) se descartan.

Conversión aproximada a grados: el centro del iris recorre ~R·sin(θ) sobre
un ojo de ancho ~30 mm con radio de globo ocular R ~12 mm. Es una estimación
geométrica sin calibración por persona: sirve para comparar dentro del mismo
sujeto, no como medida absoluta.

Limitación de fondo: a 30 fps hay una muestra cada ~33 ms y muchas sacadas
duran 20-80 ms, así que solo se detectan las sacadas grandes y su duración no
es medible con precisión. Las métricas apuntan a cambios relativos
(menos sacadas, fijaciones más largas, oscilaciones repetitivas).
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import welch

# (comisura 1, comisura 2, párpado superior, párpado inferior)
EYES = [(33, 133, 159, 145), (362, 263, 386, 374)]
IRIS_CENTERS = (468, 473)
EYE_W_MM, EYE_OPEN_MM, EYE_R_MM = 30.0, 10.0, 12.0
BLINK_EAR = 0.20
MAX_YAW, MAX_PITCH = 25.0, 20.0
PITCH_REF_MAX = 20.0
V_WEIGHT = 0.5   # peso del eje vertical en la velocidad (más ruidoso)


def _ear(pts, c1, c2, up, lo):
    w = np.linalg.norm(pts[c1] - pts[c2])
    return float(np.linalg.norm(pts[up] - pts[lo]) / w) if w > 1e-6 else 0.0


def eye_gaze(pts):
    """Devuelve (h, v, ear): h, v en [0, 1] (0.5 = centro), promedio de ambos ojos."""
    hs, vs, ears = [], [], []
    for c1, c2, up, lo in EYES:
        mid = (pts[c1] + pts[c2]) / 2
        iris = min(IRIS_CENTERS, key=lambda i: np.linalg.norm(pts[i] - mid))
        p = pts[iris]
        a, b = (pts[c1], pts[c2]) if pts[c1][0] < pts[c2][0] else (pts[c2], pts[c1])
        ax = b - a
        hs.append(float(np.dot(p - a, ax) / np.dot(ax, ax)))
        vert = pts[lo] - pts[up]
        vs.append(float(np.dot(p - pts[up], vert) / max(np.dot(vert, vert), 1e-6)))
        ears.append(_ear(pts, c1, c2, up, lo))
    return float(np.mean(hs)), float(np.mean(vs)), float(np.mean(ears))


def to_deg(h, span_mm=EYE_W_MM):
    """Ángulo aproximado (grados) a partir de una posición normalizada en [0, 1].

    span_mm es la distancia física que representa el rango 0..1: el ancho del
    ojo (~30 mm) en horizontal o la apertura palpebral (~10 mm) en vertical.
    """
    return np.degrees(np.arcsin(np.clip((np.asarray(h) - 0.5) * span_mm / EYE_R_MM, -1, 1)))


@dataclass
class Event:
    kind: str        # "fijacion" | "sacada"
    start_s: float
    end_s: float
    amp_deg: float = 0.0
    peak_vel: float = 0.0

    @property
    def duration_s(self):
        return self.end_s - self.start_s


def _smooth(s, med=5, mean=5):
    """Mediana + media móvil que respeta los huecos (NaN) de parpadeos y cuadros descartados."""
    s = np.asarray(s, dtype=float)
    bad = np.isnan(s)
    if bad.all():
        return s.copy()
    idx = np.arange(len(s))
    filled = np.interp(idx, idx[~bad], s[~bad]) if bad.any() else s
    out = uniform_filter1d(median_filter(filled, med, mode="nearest"), mean, mode="nearest")
    out[bad] = np.nan
    return out


def _runs(lab):
    """Tramos consecutivos con la misma etiqueta: [(inicio, fin, etiqueta), ...]."""
    out, i = [], 0
    while i < len(lab):
        j = i
        while j + 1 < len(lab) and lab[j + 1] == lab[i]:
            j += 1
        out.append((i, j, lab[i]))
        i = j + 1
    return out


class GazeAnalyzer:
    """Acumula la serie de mirada y calcula eventos y métricas por ventana.

    Los tiempos pueden tener paso irregular (webcam): las velocidades se
    calculan con las marcas de tiempo reales. history_s limita la memoria
    en sesiones largas (None = guardar todo, útil para procesar videos).
    """

    def __init__(self, fps=30.0, history_s=None, max_yaw=MAX_YAW, max_pitch=MAX_PITCH):
        self.fps = fps
        self.history_s = history_s
        self.max_yaw, self.max_pitch = max_yaw, max_pitch
        self.t, self.x, self.y, self.ok = [], [], [], []
        # pitch de referencia: la cámara suele quedar más abajo que la cara (en la
        # prueba con webcam, ~15° mirando de frente), así que el pitch se mide
        # respecto de la mediana de los primeros 2 s, acotada a ±PITCH_REF_MAX
        self._pitch0, self.pitch_ref = [], 0.0

    def _append(self, t_s, x, y, ok):
        self.t.append(t_s); self.x.append(x); self.y.append(y); self.ok.append(ok)
        if self.history_s and len(self.t) % 300 == 0:
            k = int(np.searchsorted(self.t, t_s - self.history_s))
            if k > 0:
                del self.t[:k], self.x[:k], self.y[:k], self.ok[:k]

    def update(self, pts, t_s, pose=None):
        """Agrega un cuadro. pose = (yaw, pitch, roll) en grados o None.

        Devuelve (h, v, ear) o None si no hay rostro.
        """
        if pts is None:
            self._append(t_s, np.nan, np.nan, False)
            return None
        h, v, ear = eye_gaze(pts)
        valid = ear >= BLINK_EAR  # en parpadeo el iris no es confiable
        if pose is not None:
            if self._pitch0 is not None:
                self._pitch0.append(pose[1])
                self.pitch_ref = float(np.clip(np.median(self._pitch0), -PITCH_REF_MAX, PITCH_REF_MAX))
                if len(self._pitch0) >= 2 * self.fps:
                    self._pitch0 = None  # referencia fija desde aquí
            if abs(pose[0]) > self.max_yaw or abs(pose[1] - self.pitch_ref) > self.max_pitch:
                valid = False     # con mucho giro la posición relativa del iris se deforma
        self._append(t_s, float(to_deg(h)), float(to_deg(v, EYE_OPEN_MM)), valid)
        return h, v, ear

    def _arrays(self, t0=None, t1=None):
        t = np.asarray(self.t, dtype=float)
        m = np.ones_like(t, bool) if t0 is None else (t >= t0) & (t < t1)
        ok = np.asarray(self.ok, dtype=bool)[m]
        # descarta también 2 cuadros alrededor de cada parpadeo
        bad = ~ok
        for k in (1, 2):
            bad[:-k] |= ~ok[k:]; bad[k:] |= ~ok[:-k]
        x = np.asarray(self.x, dtype=float)[m].copy(); y = np.asarray(self.y, dtype=float)[m].copy()
        x[bad] = np.nan; y[bad] = np.nan
        return t[m], x, y

    def ivt(self, vel_thr=30.0, min_fix_s=0.10, min_amp_deg=4.0, t0=None, t1=None):
        """Clasificación I-VT: sacada si la velocidad angular supera vel_thr (°/s).

        Con webcam el iris tiene ~0,5-1 px de ruido por cuadro (~1,5-3° con un
        ojo de 40 px), así que antes de derivar se aplica mediana de 5 + media
        de 5 cuadros, el eje vertical pesa la mitad (la apertura del ojo es de
        pocos píxeles y es el eje más ruidoso) y las "sacadas" con amplitud
        menor que min_amp_deg se reasignan a la fijación que las rodea.
        Consecuencia: solo se cuentan sacadas de unos 4° o más.
        """
        t, x, y = self._arrays(t0, t1)
        if len(t) < 3:
            return []
        # tiempos estrictamente crecientes para el gradiente
        t = np.maximum.accumulate(t) + np.arange(len(t)) * 1e-6
        dt = float(np.median(np.diff(t)))
        xs, ys = _smooth(x), _smooth(y) * V_WEIGHT
        vel = np.hypot(np.gradient(xs, t), np.gradient(ys, t))
        lab = np.where(np.isnan(vel), -1, (vel > vel_thr).astype(int))

        def amp(i, j):
            a, b = max(i - 3, 0), min(j + 3, len(xs) - 1)
            return float(np.hypot(xs[b] - xs[a], ys[b] - ys[a]))

        for i, j, v in _runs(lab):
            if v == 1 and not (amp(i, j) >= min_amp_deg):  # NaN también cuenta como ruido
                lab[i:j + 1] = 0
        events = []
        for i, j, v in _runs(lab):
            if v == 1:
                events.append(Event("sacada", t[i], t[j] + dt, amp(i, j), float(np.nanmax(vel[i:j + 1]))))
            elif v == 0 and (t[j] - t[i] + dt) >= min_fix_s:
                events.append(Event("fijacion", t[i], t[j] + dt))
        return events

    def nystagmus_index(self, band=(2.0, 6.0), t0=None, t1=None):
        """Fracción de la potencia de la mirada horizontal en una banda de oscilación.

        Una oscilación horizontal repetitiva (nistagmo) concentra energía en una
        banda estrecha. La banda es configurable; a 30 fps el límite de Nyquist
        es 15 Hz.
        La serie se remuestrea a paso uniforme porque la webcam no lo garantiza.
        """
        t, x, _ = self._arrays(t0, t1)
        good = ~np.isnan(x)
        if good.sum() < self.fps * 2 or good.mean() < 0.5:
            return np.nan
        tg = np.arange(t[good][0], t[good][-1], 1.0 / self.fps)
        if len(tg) < self.fps * 2:
            return np.nan
        xu = np.interp(tg, t[good], x[good])
        f, p = welch(xu - np.mean(xu), fs=self.fps, nperseg=min(len(xu), int(self.fps * 4)))
        tot = p[(f > 0.2)].sum()
        return float(p[(f >= band[0]) & (f <= band[1])].sum() / tot) if tot > 0 else np.nan

    def metrics(self, t0=None, t1=None, vel_thr=30.0, min_amp_deg=4.0):
        ev = self.ivt(vel_thr, min_amp_deg=min_amp_deg, t0=t0, t1=t1)
        t, x, y = self._arrays(t0, t1)
        span = (t[-1] - t[0]) if len(t) > 1 else np.nan
        fix = [e for e in ev if e.kind == "fijacion"]
        sac = [e for e in ev if e.kind == "sacada"]
        # fijación en curso: la última fijación llega hasta el final de la ventana
        en_curso = fix[-1].duration_s if fix and len(t) and fix[-1].end_s >= t[-1] else 0.0
        return {
            "validos_pct": float(np.mean(~np.isnan(x)) * 100) if len(x) else np.nan,
            "fijacion_media_s": float(np.mean([e.duration_s for e in fix])) if fix else np.nan,
            "fijacion_max_s": float(max(e.duration_s for e in fix)) if fix else np.nan,
            "fijacion_actual_s": float(en_curso),
            "fijaciones": len(fix),
            "sacadas_por_s": float(len(sac) / span) if span and span > 0 else np.nan,
            "vel_pico_media": float(np.mean([e.peak_vel for e in sac])) if sac else np.nan,
            "amp_sacada_media_deg": float(np.mean([e.amp_deg for e in sac])) if sac else np.nan,
            "dispersion_h_deg": float(np.nanstd(x)) if np.any(~np.isnan(x)) else np.nan,
            "dispersion_v_deg": float(np.nanstd(y)) if np.any(~np.isnan(y)) else np.nan,
            "indice_nistagmo": self.nystagmus_index(t0=t0, t1=t1),
        }
