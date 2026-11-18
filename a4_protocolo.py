"""Protocolo guiado de validación de A4.

Muestra un estímulo en pantalla, graba la webcam y mide cada tramo por separado:

    1. Mirada fija al punto central            -> fijación larga, ~0 sacadas
    2. Sacadas guiadas (punto salta cada 1 s)   -> ~1 sacada por segundo
    3. Seguimiento suave (punto lento)          -> pocas sacadas
    4. Mover la cabeza mirando el punto fijo    -> cuadros válidos pese al giro
    5. Punto que salta rápido (ciclo de 2 Hz)   -> oscilación horizontal
    6. Hablar / leer en voz alta                -> 0 bostezos
    7. Bostezar (o simular un bostezo largo)    -> 1 bostezo

Uso:
    python a4_protocolo.py --persona Benjamin --out protocolo/
    python a4_protocolo.py --persona Benjamin --out protocolo/ --video protocolo/Benjamin/grabacion.mp4   # re-analizar

Sale <out>/<persona>/grabacion.mp4 y <out>/<persona>/tramos.csv (métricas por tramo).
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from a4 import FaceTracker, GazeAnalyzer, YawnDetector, mar
from a4.mirada import PITCH_REF_MAX

W, H = 1280, 720
# (nombre, duración s, instrucción, posición del punto en función de t local -> (x, y) en [0, 1], esperado)
TRAMOS = [
    ("pausa", 3, "Mira el punto y manten la boca cerrada", lambda t: (0.5, 0.5), ""),
    ("fija", 10, "1/7 Mira el punto sin mover los ojos", lambda t: (0.5, 0.5), "sacadas ~0/s"),
    ("sacadas", 12, "2/7 Sigue el punto cuando salte", lambda t: (0.15 if int(t) % 2 else 0.85, 0.5),
     "sacadas ~1/s"),
    ("seguimiento", 12, "3/7 Sigue el punto con la vista",
     lambda t: (0.5 + 0.35 * np.sin(2 * np.pi * 0.15 * t), 0.5), "sacadas bajas"),
    ("cabeza", 10, "4/7 Mira el punto y mueve la cabeza a los lados", lambda t: (0.5, 0.5), "sacadas bajas"),
    # el punto alterna cada 0,25 s (ciclo de 2 Hz, dentro de la banda del índice de nistagmo),
    # porque a voluntad los ojos no oscilan tan rápido
    ("oscilacion", 8, "5/7 Sigue el punto que salta rapido", lambda t: (0.40 if int(t * 4) % 2 else 0.60, 0.5),
     "nistagmo alto"),
    ("habla", 10, "6/7 Habla o lee en voz alta", lambda t: (0.5, 0.5), "0 bostezos"),
    ("bostezo", 10, "7/7 Bosteza (o abre grande la boca 3-4 s)", lambda t: (0.5, 0.5), "1 bostezo"),
]


def estimulo(instr, pos, t_loc, dur):
    img = np.full((H, W, 3), 30, np.uint8)
    if pos is None:  # oscilación: dos marcas fijas
        for x in (0.42, 0.58):
            cv2.circle(img, (int(x * W), H // 2), 14, (0, 200, 255), -1)
    else:
        x, y = pos(t_loc)
        cv2.circle(img, (int(x * W), int(y * H)), 16, (0, 255, 0), -1)
        cv2.circle(img, (int(x * W), int(y * H)), 4, (0, 0, 0), -1)
    cv2.putText(img, instr, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"{dur - t_loc:.0f} s", (W - 120, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    return img


def tramo_en(t):
    t0 = 0.0
    for tr in TRAMOS:
        if t < t0 + tr[1]:
            return tr, t - t0
        t0 += tr[1]
    return None, 0.0


def medir(muestras, fps, pitch_ref=None):
    """muestras: lista de (t, pts|None, pose) de un tramo -> métricas de mirada y bostezos.

    pitch_ref: pitch de la postura frontal (medido en la pausa inicial); si no se da,
    cada tramo lo estima con sus primeros 2 s.
    """
    g, y = GazeAnalyzer(fps), YawnDetector(fps=fps)
    if pitch_ref is not None:
        g.pitch_ref, g._pitch0 = pitch_ref, None
    for t, pts, pose in muestras:
        g.update(pts, t, pose)
        if pts is not None:
            y.update(mar(pts), t)
    m = g.metrics()
    m["bostezos"] = len(y.finish(muestras[-1][0])) if muestras else 0
    m["mar_max"] = float(np.nanmax([mar(p) for _, p, _ in muestras if p is not None] or [np.nan]))
    return m


def esperar_inicio(cap):
    """Vista previa de la cámara hasta que se presione ESPACIO (True) o Q/ESC (False)."""
    while True:
        ok, frame = cap.read()
        if not ok:
            return False
        img = np.full((H, W, 3), 30, np.uint8)
        prev = cv2.resize(frame, (640, 480))
        img[120:600, (W - 640) // 2:(W + 640) // 2] = prev
        cv2.putText(img, "Centra la cara, buena luz, a ~60 cm. ESPACIO = empezar, Q = salir",
                    (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Protocolo A4", img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            return True
        if k in (27, ord("q")):
            return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", required=True)
    ap.add_argument("--out", default="protocolo")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--video", help="re-analiza una grabación previa en vez de usar la webcam")
    a = ap.parse_args()
    out = Path(a.out) / a.persona
    out.mkdir(parents=True, exist_ok=True)

    live = a.video is None
    cap = cv2.VideoCapture(a.cam if live else a.video)
    if not cap.isOpened():
        raise SystemExit("[ERROR] No se pudo abrir la cámara o el video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 < fps <= 240 else 30.0
    tracker, writer = FaceTracker(), None
    por_tramo = {tr[0]: [] for tr in TRAMOS}
    total = sum(tr[1] for tr in TRAMOS)
    if live:
        cv2.namedWindow("Protocolo A4", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Protocolo A4", W, H)
        if not esperar_inicio(cap):
            cap.release(); tracker.close(); cv2.destroyAllWindows()
            raise SystemExit("[INFO] Protocolo cancelado")
    t_start, i = time.monotonic(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = (time.monotonic() - t_start) if live else i / fps
        i += 1
        if t >= total:
            break
        tr, t_loc = tramo_en(t)
        face = tracker.process(frame, t * 1000)
        por_tramo[tr[0]].append((t, face.pts if face else None, face.pose if face else None))
        if live:
            if writer is None:
                writer = cv2.VideoWriter(str(out / "grabacion.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (frame.shape[1], frame.shape[0]))
            writer.write(frame)
            img = estimulo(tr[2], tr[3], t_loc, tr[1])
            # sin miniatura de la cámara para que no atraiga la mirada
            if face is None:
                cv2.putText(img, "SIN ROSTRO", (40, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow("Protocolo A4", img)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    cap.release(); tracker.close()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    filas = []
    pitches = [p[1] for _, _, p in por_tramo["pausa"] if p is not None]
    pitch_ref = float(np.clip(np.median(pitches), -PITCH_REF_MAX, PITCH_REF_MAX)) if pitches else None
    for nombre, _, _, _, esperado in TRAMOS:
        if nombre == "pausa" or not por_tramo[nombre]:
            continue
        m = medir(por_tramo[nombre], fps, pitch_ref)
        filas.append({"persona": a.persona, "tramo": nombre, "esperado": esperado,
                      "sacadas_por_s": m["sacadas_por_s"], "fijacion_media_s": m["fijacion_media_s"],
                      "amp_sacada_media_deg": m["amp_sacada_media_deg"], "indice_nistagmo": m["indice_nistagmo"],
                      "bostezos": m["bostezos"], "mar_max": m["mar_max"], "validos_pct": m["validos_pct"]})
    df = pd.DataFrame(filas)
    df.to_csv(out / "tramos.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.round(2).to_string(index=False))
    print(f"\n[OK] Resultados en {out}")


if __name__ == "__main__":
    main()
