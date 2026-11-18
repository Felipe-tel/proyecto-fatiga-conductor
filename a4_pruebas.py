"""Pruebas sintéticas de A4: verifican la lógica de bostezos y mirada con
landmarks generados (apertura de boca, posición del iris, párpados, movimiento
de cabeza, ruido y fps irregulares controlados).

Uso:
    python a4_pruebas.py            # imprime la tabla de resultados y termina con error si alguna falla
"""
import numpy as np

from a4 import A4Monitor, GazeAnalyzer, YawnDetector, mar

RNG = np.random.default_rng(0)
FPS = 30.0


def cara(boca=0.0, h=0.5, v=0.5, ear=0.30, dx=0.0, dy=0.0, ruido=0.0):
    """Landmarks (478, 2) sintéticos. boca = MAR deseado; h, v = posición del iris en el ojo."""
    p = np.tile([320.0, 240.0], (478, 1))
    for c1, c2, up, lo, iris, x0 in [(33, 133, 159, 145, 468, 250), (362, 263, 386, 374, 473, 350)]:
        p[c1], p[c2] = (x0, 200), (x0 + 40, 200)
        p[up], p[lo] = (x0 + 20, 200 - ear * 20), (x0 + 20, 200 + ear * 20)
        p[iris] = (x0 + h * 40, p[up][1] + v * (p[lo][1] - p[up][1]))
    p[78], p[308] = (290, 320), (350, 320)
    for i_up, i_lo, x in [(81, 178, 305), (13, 14, 320), (311, 402, 335)]:
        p[i_up], p[i_lo] = (x, 320 - boca * 30), (x, 320 + boca * 30)
    p += (dx, dy)
    if ruido:
        p += RNG.normal(0, ruido, p.shape)
    return p.astype(np.float32)


def tiempos(dur, fps=FPS, jitter=False, t0=0.0):
    if not jitter:
        return t0 + np.arange(int(dur * fps)) / fps
    return t0 + np.cumsum(RNG.uniform(1 / 30, 1 / 15, int(dur * 25)))


resultados = []


def check(nombre, esperado, obtenido, ok):
    resultados.append((nombre, esperado, obtenido, bool(ok)))


# ---------------------------------------------------------------- bostezos
check("MAR boca cerrada / abierta", "0.00 / 0.80", f"{mar(cara(0)):.2f} / {mar(cara(0.8)):.2f}",
      mar(cara(0)) < 0.01 and abs(mar(cara(0.8)) - 0.8) < 1e-3)


def bostezos(serie, **kw):
    det = YawnDetector(fps=FPS, **kw)
    for t, m in serie:
        det.update(m, t)
    return det.finish(serie[-1][0])


def tramo(t0, dur, valor):
    return [(t0 + i / FPS, valor) for i in range(int(dur * FPS))]


s = tramo(0, 2, 0.02) + tramo(2, 3, 0.75) + tramo(5, 2, 0.02) + tramo(7, 0.5, 0.75) + tramo(7.5, 2, 0.02)
ev = bostezos(s)
check("Bostezo 3 s aceptado, apertura 0.5 s descartada", "1 evento de ~3 s",
      f"{len(ev)} evento(s) " + ", ".join(f"{e.duration_s:.1f} s" for e in ev),
      len(ev) == 1 and 2.8 < ev[0].duration_s < 3.3)

habla = [(i / FPS, 0.6 if (i // 4) % 2 else 0.05) for i in range(int(6 * FPS))]  # ~4 Hz durante 6 s
check("Hablar (aperturas de 4 Hz, 6 s)", "0 bostezos", f"{len(bostezos(habla))}", len(bostezos(habla)) == 0)

T0 = 1.7e9  # tiempos absolutos como time.time()
s = [(T0 + t, 0.02 + RNG.normal(0, 0.005)) for t, _ in tramo(0, 6, 0)] + \
    [(T0 + t, v) for t, v in tramo(6, 3, 0.7) + tramo(9, 2, 0.02)]
det = YawnDetector(fps=FPS, calib_s=5)
for t, m in s:
    det.update(m, t)
ev = det.finish(s[-1][0])
check("Calibración 5 s con tiempos absolutos", "umbral >= 0.40 (piso) y 1 bostezo",
      f"umbral {det.thr_on:.2f}, {len(ev)} bostezo(s)", det.thr_on >= 0.4 and len(ev) == 1)

s = [(t, 0.6 if (int(t * 8) % 2) else 0.05) for t, _ in tramo(0, 5, 0)] + tramo(5, 2, 0.02) + \
    tramo(7, 3, 0.8) + tramo(10, 2, 0.02)  # habla durante la calibración y luego bosteza
det = YawnDetector(fps=FPS, calib_s=5)
for t, m in s:
    det.update(m, t)
ev = det.finish(s[-1][0])
check("Habla durante la calibración", "umbral con techo 0.70, 1 bostezo",
      f"umbral {det.thr_on:.2f}, {len(ev)} bostezo(s)", det.thr_on <= 0.70 and len(ev) == 1)

# ---------------------------------------------------------------- mirada


def mirada(gen, dur=10, jitter=False, pose=None, ruido=0.5):
    g = GazeAnalyzer(FPS)
    for t in tiempos(dur, jitter=jitter):
        g.update(cara(ruido=ruido, **gen(t)), t, pose)
    return g.metrics()


m = mirada(lambda t: {})
check("Mirada fija 10 s (ruido 0.5 px)", "~0 sacadas, fijación larga",
      f"{m['sacadas_por_s']:.2f} sac/s, fij. media {m['fijacion_media_s']:.1f} s",
      m["sacadas_por_s"] < 0.15 and m["fijacion_media_s"] > 3)

m = mirada(lambda t: {}, ruido=1.0)
check("Mirada fija 10 s (ruido 1 px)", "< 0.3 sacadas falsas/s", f"{m['sacadas_por_s']:.2f} sac/s",
      m["sacadas_por_s"] < 0.3)

m = mirada(lambda t: {"h": 0.42 if int(t) % 2 else 0.58})
check("Sacadas guiadas ±11° cada 1 s", "~1 sac/s",
      f"{m['sacadas_por_s']:.2f} sac/s, amp {m['amp_sacada_media_deg']:.0f}°",
      0.8 <= m["sacadas_por_s"] <= 1.1)

m = mirada(lambda t: {"h": 0.42 if int(t) % 2 else 0.58}, jitter=True)
check("Sacadas guiadas con fps irregular (15-30)", "~1 sac/s", f"{m['sacadas_por_s']:.2f} sac/s",
      0.8 <= m["sacadas_por_s"] <= 1.1)

m = mirada(lambda t: {"dx": 60 * np.sin(2 * np.pi * 0.5 * t), "dy": 20 * np.sin(2 * np.pi * 0.3 * t)})
check("Cabeza trasladándose ±60 px, mirada fija", "~0 sacadas", f"{m['sacadas_por_s']:.2f} sac/s",
      m["sacadas_por_s"] < 0.15)

m = mirada(lambda t: {"ear": 0.05 if (t % 3) < 0.2 else 0.30})
check("Parpadeos cada 3 s", "sin sacadas falsas, válidos < 100 %",
      f"{m['sacadas_por_s']:.2f} sac/s, válidos {m['validos_pct']:.0f} %",
      m["sacadas_por_s"] < 0.15 and m["validos_pct"] < 95)

m = mirada(lambda t: {}, pose=(40.0, 0.0, 0.0))
check("Cabeza girada 40° (yaw)", "cuadros descartados", f"válidos {m['validos_pct']:.0f} %", m["validos_pct"] == 0)

g = GazeAnalyzer(FPS)  # cámara más abajo que la cara: pitch ~15° de frente (2 s quieto), luego ±8°
for t in tiempos(10):
    g.update(cara(ruido=0.5), t, (0.0, 15.0 + (8 * np.sin(2 * np.pi * 0.3 * (t - 2)) if t >= 2 else 0), 0.0))
m = g.metrics()
check("Cámara baja (pitch ~15° de frente ±8°)", "cuadros válidos", f"válidos {m['validos_pct']:.0f} %, "
      f"pitch ref {g.pitch_ref:.0f}°", m["validos_pct"] > 95)

base = mirada(lambda t: {})["indice_nistagmo"]
osc = mirada(lambda t: {"h": 0.5 + 0.04 * np.sin(2 * np.pi * 3 * t)})["indice_nistagmo"]
check("Oscilación horizontal de 3 Hz", "índice alto vs mirada fija", f"{osc:.2f} vs {base:.2f}",
      osc > 0.6 and osc > base + 0.3)

# ---------------------------------------------------------------- monitor completo
mon = A4Monitor(fps=FPS, calib_s=5, baseline_s=30, yawn_alert_n=3)
niveles, alertas = [], set()
for t in tiempos(120):
    if t < 35:      # calibración + línea base: explora con sacadas cada 1 s
        kw = {"h": 0.42 if int(t) % 2 else 0.58}
    elif t < 65:    # mirada fija (exploración reducida)
        kw = {}
    else:           # tres bostezos de 4 s
        kw = {"boca": 0.7 if any(a <= t < a + 4 for a in (70, 85, 100)) else 0.02}
        kw["h"] = 0.42 if int(t) % 2 else 0.58
    st = mon.update(cara(ruido=0.5, **kw), T0 + t)
    niveles.append(st.nivel); alertas.update(a.split(" ")[0] for a in st.alertas)
res = mon.finish()
check("Monitor: 3 bostezos en 2 min", "3 bostezos, alerta FATIGA",
      f"{len(res['bostezos'])} bostezos, alertas {sorted(alertas)}",
      len(res["bostezos"]) == 3 and "FATIGA:" in alertas)
check("Monitor: mirada fija tras la línea base", "MIRADA FIJA y EXPLORACION VISUAL REDUCIDA",
      ", ".join(sorted(alertas)), {"MIRADA", "EXPLORACION"} <= alertas)
check("Monitor: ventanas de 10 s para A6", "12 filas", f"{len(mon.rows)} filas, línea base "
      f"{'lista' if res['linea_base_mirada'] else 'no'}", len(mon.rows) == 12 and res["linea_base_mirada"])

mon = A4Monitor(fps=FPS, calib_s=5, baseline_s=30)
alertas = set()
for t in tiempos(90):  # conducta constante: sacadas cada ~1.5 s toda la sesión
    st = mon.update(cara(ruido=0.5, h=0.42 if int(t / 1.5) % 2 else 0.58), t)
    alertas.update(st.alertas)
mon.finish()
rel = [r["fijacion_media_rel"] for r in mon.rows if r["linea_base"]]
check("Monitor: conducta constante", "cocientes ~1 y sin alertas",
      f"fij. rel {np.nanmin(rel):.2f}-{np.nanmax(rel):.2f}, alertas: {sorted(alertas) or 'ninguna'}",
      not alertas and 0.7 <= np.nanmin(rel) and np.nanmax(rel) <= 1.4)

# ---------------------------------------------------------------- reporte
w = max(len(r[0]) for r in resultados)
print(f"{'Prueba':<{w}} | {'Esperado':<38} | {'Obtenido':<45} | OK")
print("-" * (w + 95))
for n, e, o, ok in resultados:
    print(f"{n:<{w}} | {e:<38} | {o:<45} | {'si' if ok else 'NO'}")
fallas = [r for r in resultados if not r[3]]
print(f"\n{len(resultados) - len(fallas)}/{len(resultados)} pruebas correctas")
raise SystemExit(1 if fallas else 0)
