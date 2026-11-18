"""Evaluación del detector de bostezos sobre YawDD con etiquetas a nivel de video.

YawDD nombra los videos del espejo retrovisor como "<id>-<Sexo><Lentes>-<Normal|Talking|Yawning>.avi".
Los id se repiten entre hombres y mujeres, y la misma persona aparece con y sin
lentes, así que el sujeto es "<id>-<Sexo>". Un video es positivo si su nombre
contiene "Yawning" y la predicción es positiva si se detecta al menos un bostezo.

Modos:
    --thr/--min-dur fijos      evalúa una configuración sobre todos los videos
    --sweep                    barrido de umbrales sobre todos los videos (análisis de sensibilidad;
                               el resultado de generalización es el de --cv)
    --cv K                     validación cruzada por sujeto: en cada pliegue el umbral y la
                               duración se eligen con los sujetos de entrenamiento y se miden
                               en los de prueba (este es el número que va al informe)
    --test 11-Female,...       división fija: el umbral se elige con los demás sujetos y se mide
                               en estos (la misma división que evaluate_yawdd_final_test.py)

Las series de MAR se guardan en <out>/mar_cache.npz para no reprocesar los videos.

Uso:
    python a4_eval_yawdd.py --root ruta/YawDD --out eval/ --cv 5
    python a4_eval_yawdd.py --root data/raw/yawdd --out eval/ --test 11-Female,12-Female,17-Male,18-Male
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from a4 import FaceTracker, mar
from a4.boca import YawnDetector

GRID = [(round(t, 2), d) for t in np.arange(0.30, 0.81, 0.05) for d in (0.5, 1.0, 1.5, 2.0, 3.0)]


def mar_series(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tr, vals, i = FaceTracker(), [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        face = tr.process(f, i / fps * 1000)
        vals.append(mar(face.pts) if face is not None else np.nan)
        i += 1
    cap.release(); tr.close()
    return fps, np.array(vals)


CALIB_S = 0.0  # --calib: simula la calibración del panel en los primeros segundos de cada video


def detect(fps, series, thr, min_dur):
    det = YawnDetector(fps=fps, thr_on=thr, thr_off=thr * 0.8, min_dur_s=min_dur, calib_s=CALIB_S)
    for i, v in enumerate(series):
        if not np.isnan(v):
            det.update(v, i / fps)
    return det.finish(len(series) / fps)


def label(name):
    m = re.search(r"(Normal|Talking|Yawning)", name, re.I)
    return m.group(1).capitalize() if m else None


def subject(name):
    m = re.match(r"(\d+)-(Female|Male)", name, re.I)
    return f"{m.group(1)}-{m.group(2).capitalize()}" if m else name.split("-")[0]


def scores(df):
    tp = ((df.pred == 1) & (df.y == 1)).sum(); fp = ((df.pred == 1) & (df.y == 0)).sum()
    fn = ((df.pred == 0) & (df.y == 1)).sum()
    p = tp / (tp + fp) if tp + fp else np.nan; r = tp / (tp + fn) if tp + fn else np.nan
    talking = df[df.clase == "Talking"]
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p and r else np.nan,
            "fp_hablando": int(((df.pred == 1) & (df.clase == "Talking")).sum()),
            "fp_hablando_pct": float(talking.pred.mean() * 100) if len(talking) else np.nan,
            "videos": len(df)}


def predict(cache, names, thr, dur):
    rows = []
    for name in names:
        fps, s = cache[name]
        ev = detect(fps, s, thr, dur)
        c = label(name)
        rows.append({"video": name, "sujeto": subject(name), "clase": c, "y": int(c == "Yawning"),
                     "pred": int(len(ev) > 0), "n_bostezos": len(ev), "mar_p95": float(np.nanpercentile(s, 95))
                     if np.any(~np.isnan(s)) else np.nan})
    return pd.DataFrame(rows)


def best_params(cache, names):
    res = [(scores(predict(cache, names, t, d))["f1"], -d, t, d) for t, d in GRID]
    res = [r for r in res if not np.isnan(r[0])]
    _, _, t, d = max(res)  # mejor F1; en empate, la duración más corta
    return t, d


def load_cache(vids, cache_file):
    cache = {}
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=True)
        cache = {k: (float(v[0]), v[1]) for k, v in z["data"].item().items()}
    todo = [p for p in vids if p.name not in cache]
    for k, p in enumerate(todo, 1):
        print(f"[{k}/{len(todo)}] {p.name}")
        cache[p.name] = mar_series(p)
        if k % 10 == 0 or k == len(todo):
            np.savez(cache_file, data=np.array({n: (f, s) for n, (f, s) in cache.items()}, dtype=object))
    return cache


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="eval")
    ap.add_argument("--thr", type=float, default=0.4)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--cv", type=int, default=0, help="número de pliegues por sujeto")
    ap.add_argument("--test", default=None, help="sujetos de prueba separados por coma (ej. 11-Female,17-Male)")
    ap.add_argument("--calib", type=float, default=0.0, help="segundos de calibración por video, como en el panel")
    a = ap.parse_args()
    CALIB_S = a.calib

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    vids = sorted(p for p in Path(a.root).rglob("*") if p.suffix.lower() in (".avi", ".mp4") and label(p.name))
    print(f"{len(vids)} videos con etiqueta, {len({subject(p.name) for p in vids})} sujetos")
    cache = load_cache(vids, out / "mar_cache.npz")
    names = [p.name for p in vids]

    if a.test:
        test_subs = {subject(s.strip() + "-") for s in a.test.split(",")}
        test = [n for n in names if subject(n) in test_subs]
        train = [n for n in names if subject(n) not in test_subs]
        if not test or not train:
            raise SystemExit(f"Se necesitan videos de entrenamiento y de prueba (prueba: {len(test)}, entrenamiento: {len(train)})")
        thr, dur = best_params(cache, train)
        df = predict(cache, test, thr, dur).assign(umbral=thr, min_dur_s=dur)
        df.to_csv(out / "test_por_video.csv", index=False)
        print(f"Entrenamiento: {len({subject(n) for n in train})} sujetos -> umbral {thr}, min_dur {dur} s")
        print(df[["video", "clase", "pred", "n_bostezos", "mar_p95"]].to_string(index=False))
        print("\nResultado en prueba:", {k: round(float(v), 3) if isinstance(v, float) else v
                                        for k, v in scores(df).items()},
              f"| aciertos {int((df.pred == df.y).sum())}/{len(df)}")
    elif a.cv:
        subs = sorted({subject(n) for n in names})
        rng = np.random.default_rng(0); rng.shuffle(subs)
        fold_of = {s: i % a.cv for i, s in enumerate(subs)}
        preds, folds = [], []
        for k in range(a.cv):
            train = [n for n in names if fold_of[subject(n)] != k]
            test = [n for n in names if fold_of[subject(n)] == k]
            thr, dur = best_params(cache, train)
            df = predict(cache, test, thr, dur).assign(pliegue=k, umbral=thr, min_dur_s=dur)
            preds.append(df)
            folds.append({"pliegue": k, "umbral": thr, "min_dur_s": dur, **scores(df)})
            print(f"pliegue {k}: umbral {thr}, min_dur {dur} s -> F1 {folds[-1]['f1']:.3f}")
        allp = pd.concat(preds)
        allp.to_csv(out / "cv_por_video.csv", index=False)
        pd.DataFrame(folds).to_csv(out / "cv_pliegues.csv", index=False)
        tot = scores(allp)
        print("\nResultado en prueba (todos los pliegues):",
              {k: round(float(v), 3) if isinstance(v, float) else v for k, v in tot.items()})
    else:
        grid = GRID if a.sweep else [(a.thr, a.min_dur)]
        res = []
        for thr, dur in grid:
            df = predict(cache, names, thr, dur)
            res.append({"umbral": thr, "min_dur_s": dur, **scores(df)})
            if not a.sweep:
                df.to_csv(out / "por_video.csv", index=False)
        r = pd.DataFrame(res).sort_values("f1", ascending=False)
        r.to_csv(out / ("barrido.csv" if a.sweep else "resumen.csv"), index=False)
        print(r.head(10).to_string(index=False))
