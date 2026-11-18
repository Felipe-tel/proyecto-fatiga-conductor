"""Procesa un video (o la webcam) solo con el módulo A4 y guarda indicadores y resumen.

Uso:
    python a4_run_video.py --video ruta/al/video.avi --out resultados/ --overlay
    python a4_run_video.py --cam 0 --show --calib 5
"""
import argparse
import time
from pathlib import Path

import cv2

from a4 import A4Monitor, FaceTracker


def process(source, out_dir=None, show=False, calib_s=0.0, window_s=10.0, overlay=False, baseline_s=30.0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] No se pudo abrir la fuente: {source}")
    live = isinstance(source, int)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1 or fps > 240:
        fps = 30.0
    tracker = FaceTracker()
    mon = A4Monitor(fps=fps, calib_s=calib_s, window_s=window_s, baseline_s=baseline_s,
                    history_s=120.0 if live else None, keep_frames=True)
    writer, i, t_start = None, 0, time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # en video el tiempo sale del número de cuadro; en vivo, del reloj
        t = (time.monotonic() - t_start) if live else i / fps
        face = tracker.process(frame, t * 1000)
        st = mon.update(face.pts if face else None, t, face.pose if face else None,
                        face.blendshapes if face else None)
        if show or overlay:
            mon.draw(frame, st, face.pts if face else None, origin=(10, 30))
        if overlay and out_dir:
            if writer is None:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(Path(out_dir) / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        if show:
            cv2.imshow("A4", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
        i += 1
    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    tracker.close()
    resumen = mon.finish()
    resumen["fps"] = fps
    if out_dir:
        mon.save(out_dir, resumen)
    return mon, resumen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--cam", type=int)
    ap.add_argument("--out", default="resultados")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--overlay", action="store_true", help="guarda overlay.mp4 con los landmarks y el estado A4")
    ap.add_argument("--calib", type=float, default=0.0, help="segundos iniciales de calibración (boca en reposo)")
    ap.add_argument("--window", type=float, default=10.0, help="largo de la ventana de indicadores (s)")
    ap.add_argument("--baseline", type=float, default=30.0, help="segundos de línea base de la mirada")
    a = ap.parse_args()
    src = a.video if a.video else (a.cam if a.cam is not None else 0)
    _, s = process(src, a.out, a.show, a.calib, a.window, a.overlay, a.baseline)
    print(f"Bostezos detectados: {len(s['bostezos'])} (umbral MAR {s['umbral_mar']:.2f})")
    for e in s["bostezos"]:
        print(f"  {e['inicio_s']:.1f}-{e['fin_s']:.1f} s  (MAR pico {e['mar_pico']:.2f})")
    print("Mirada:", {k: round(v, 3) if isinstance(v, float) else v for k, v in s["mirada_global"].items()})
    print(f"Ventanas: {s['ventanas']} ({s['ventanas_con_alerta']} con alerta) -> {a.out}")
