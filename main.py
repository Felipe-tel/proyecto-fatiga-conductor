import argparse
import csv
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from pathlib import Path

try:
    import winsound
    SONIDO_DISPONIBLE = True
except ImportError:
    SONIDO_DISPONIBLE = False

from face_landmarks import (
    BASE_DIR, MODEL_PATH, TIEMPO_MICROSUEÑO_SEG, SCALE_FACTOR,
    LEFT_EYE, RIGHT_EYE, LEFT_EYE_EAR, RIGHT_EYE_EAR, LEFT_IRIS, RIGHT_IRIS, MOUTH,
    obtener_puntos, dibujar_puntos, dibujar_contorno, obtener_roi, dibujar_caja
)
from pupil_segmentation import (
    preprocess_eye_roi, segment_pupil, calcular_ear, calcular_diametro_iris, calcular_pir
)
from head_pose import (
    obtener_puntos_pose_2d, estimar_pose_cabeza, dibujar_vector_pose, evaluar_estado_pose
)
from driver_state_classifier import ClasificadorEstadoConductor
from a4 import A4Monitor, landmarks_to_array, blendshapes_to_dict, pose_from_result

# ==============================================================================
# PARÁMETROS CALIBRADOS
# ==============================================================================
UMBRAL_EAR_CERRADO = 0.18       # Ojo cerrado real (tolera entrecerrar ojos al sonreír)
# El bostezo (MAR, umbral calibrado y duración mínima) lo calcula el módulo A4 (a4/boca.py)

UMBRAL_PITCH_CABECEO = 18.0     # Cabeceo hacia adelante/abajo (en grados)
UMBRAL_YAW_DISTRACCION = 32.0   # Giro lateral izquierda/derecha (en grados)
UMBRAL_ROLL_LATERAL = 22.0      # Inclinación hacia los hombros (en grados)


def registrar_prueba_a7(carpeta, num_prueba, img_name, estado_global, ear, perclos, mar, pir, pitch, yaw):
    archivo_csv = carpeta / "registro_pruebas_a7.csv"
    existe = archivo_csv.exists()
    with open(archivo_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow([
                "Prueba", "Timestamp", "Estado_Clasificado",
                "EAR", "PERCLOS_%", "MAR", "PIR", "Pitch_deg", "Yaw_deg", "Imagen"
            ])
        writer.writerow([
            num_prueba,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            estado_global,
            f"{ear:.2f}",
            f"{perclos:.1f}",
            f"{mar:.2f}",
            f"{pir:.2f}" if pir is not None else "",
            f"{pitch:.1f}" if pitch is not None else "",
            f"{yaw:.1f}" if yaw is not None else "",
            img_name
        ])


def main():
    parser = argparse.ArgumentParser(description="Sistema Integrado de Monitoreo en Cabina (TEL-328)")
    parser.add_argument("--video", type=str, default=None, help="Ruta a video de prueba (opcional)")
    parser.add_argument("--cam", type=int, default=0, help="Índice de la cámara si no se usa --video (prueba 1, 2... si abre otra)")
    parser.add_argument("--calib-a4", type=float, default=5.0, help="Segundos iniciales con la boca cerrada para calibrar el bostezo (A4)")
    parser.add_argument("--salida", type=str, default=None, help="Carpeta donde guardar los indicadores de A4 al cerrar")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"[ERROR] No se encontró el modelo en: {MODEL_PATH}")
        return

    fuente = args.video if args.video else args.cam
    es_video = args.video is not None
    cap = cv2.VideoCapture(fuente)

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la fuente de video: {fuente}")
        return

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
        output_face_blendshapes=True,                # A4: jawOpen
        output_facial_transformation_matrixes=True   # A4: pose para descartar cuadros girados
    )

    historial_perclos = deque(maxlen=90)
    trayectoria_iris = deque(maxlen=30)
    clasificador_a6 = ClasificadorEstadoConductor(ventana_frames=45)
    monitor_a4 = A4Monitor(calib_s=args.calib_a4, history_s=None if args.video else 120.0)

    inicio_cierre_ojos = None
    duracion_cierre = 0.0
    ojo_estaba_cerrado = False
    bostezo_activo = False
    pir_suavizado = None

    mostrar_rois = True
    contador_evidencias = 0
    tiempo_anterior = time.time()
    fps_suavizado = 0.0
    ultimo_timestamp = 0
    ultimo_beep = 0.0

    carpeta_evidencias = BASE_DIR / "evidencias"
    carpeta_evidencias.mkdir(exist_ok=True)

    print(f"[OK] Pruebas y capturas se guardarán en: {carpeta_evidencias}")
    print("Controles: [S] Guardar prueba A7 | [V] Mostrar/Ocultar ROIs pupila | [Q] Salir")

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if not es_video:
                frame = cv2.flip(frame, 1)

            frame_limpio = frame.copy()
            alto, ancho = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame_limpio, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            if es_video:
                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                tiempo_seg = timestamp_ms / 1000.0
            else:
                timestamp_ms = int(time.monotonic() * 1000)
                tiempo_seg = time.time()

            if timestamp_ms <= ultimo_timestamp:
                timestamp_ms = ultimo_timestamp + 1
            ultimo_timestamp = timestamp_ms

            resultado = landmarker.detect_for_video(mp_image, timestamp_ms)

            # A4: bostezos (MAR calibrado, validado en YawDD) y mirada (fijaciones/sacadas)
            pts_a4 = landmarks_to_array(resultado.face_landmarks[0], ancho, alto) if resultado.face_landmarks else None
            estado_a4 = monitor_a4.update(pts_a4, tiempo_seg, pose=pose_from_result(resultado),
                                          blendshapes=blendshapes_to_dict(resultado))

            ear_promedio = 0.0
            perclos_actual = 0.0
            mar_actual = 0.0
            sacadico_actual = 0.0
            pitch, yaw, roll = None, None, None
            estado_pose = "SIN ROSTRO"
            estado_global = "ROSTRO NO DETECTADO"
            color_global = (0, 0, 255)
            score_f, score_s = 0.0, 0.0
            alerta_critica = False

            roi_ojo_izq, box_ojo_izq = None, None
            roi_ojo_der, box_ojo_der = None, None
            iris_izq_pts, iris_der_pts = [], []

            if resultado.face_landmarks:
                landmarks = resultado.face_landmarks[0]

                ojo_izq = obtener_puntos(landmarks, LEFT_EYE, ancho, alto)
                ojo_der = obtener_puntos(landmarks, RIGHT_EYE, ancho, alto)
                iris_izq_pts = obtener_puntos(landmarks, LEFT_IRIS, ancho, alto)
                iris_der_pts = obtener_puntos(landmarks, RIGHT_IRIS, ancho, alto)
                boca = obtener_puntos(landmarks, MOUTH, ancho, alto)

                # 1. EAR y PERCLOS (A3)
                ear_izq = calcular_ear(obtener_puntos(landmarks, LEFT_EYE_EAR, ancho, alto))
                ear_der = calcular_ear(obtener_puntos(landmarks, RIGHT_EYE_EAR, ancho, alto))
                ear_promedio = (ear_izq + ear_der) / 2.0

                historial_perclos.append(1 if ear_promedio < UMBRAL_EAR_CERRADO else 0)
                perclos_actual = (sum(historial_perclos) / len(historial_perclos)) * 100.0

                if ear_promedio < UMBRAL_EAR_CERRADO:
                    if not ojo_estaba_cerrado:
                        ojo_estaba_cerrado = True
                        inicio_cierre_ojos = tiempo_seg
                    duracion_cierre = tiempo_seg - inicio_cierre_ojos
                else:
                    ojo_estaba_cerrado = False
                    inicio_cierre_ojos = None
                    duracion_cierre = 0.0

                # 2. MAR e inestabilidad sacádica (A4)
                # el bostezo lo decide el módulo A4 (umbral calibrado por persona + duración mínima)
                mar_actual = estado_a4.mar_suav
                bostezo_activo = estado_a4.bostezando

                if iris_izq_pts and len(ojo_izq) >= 9:
                    p_iris = np.array(iris_izq_pts[0], dtype=np.float32)
                    ancho_ojo = np.linalg.norm(np.array(ojo_izq[0]) - np.array(ojo_izq[8])) + 1e-5
                    trayectoria_iris.append(p_iris / ancho_ojo)
                    if len(trayectoria_iris) >= 4:
                        diffs = [np.linalg.norm(trayectoria_iris[i] - trayectoria_iris[i - 1]) for i in range(1, len(trayectoria_iris))]
                        sacadico_actual = float(np.std(diffs) * 100.0)

                # 3. Pose 3D de cabeza (A5)
                puntos_pose_2d = obtener_puntos_pose_2d(landmarks, ancho, alto)
                angulos_pose, vector_nariz, _ = estimar_pose_cabeza(puntos_pose_2d, ancho, alto)
                if angulos_pose is not None and vector_nariz is not None:
                    pitch, yaw, roll = angulos_pose
                    estado_pose, _ = evaluar_estado_pose(
                        pitch, yaw, roll,
                        pitch_umbral=UMBRAL_PITCH_CABECEO,
                        yaw_umbral=UMBRAL_YAW_DISTRACCION,
                        roll_umbral=UMBRAL_ROLL_LATERAL
                    )
                    dibujar_vector_pose(frame, vector_nariz[0], vector_nariz[1], puntos_2d=puntos_pose_2d)

                dibujar_contorno(frame, ojo_izq, (255, 0, 0))
                dibujar_contorno(frame, ojo_der, (255, 0, 0))
                dibujar_puntos(frame, iris_izq_pts, (0, 255, 255), 2)
                dibujar_puntos(frame, iris_der_pts, (0, 255, 255), 2)
                dibujar_contorno(frame, boca, (255, 0, 255))

                roi_ojo_izq, box_ojo_izq = obtener_roi(frame_limpio, ojo_izq, margen=15)
                roi_ojo_der, box_ojo_der = obtener_roi(frame_limpio, ojo_der, margen=15)
                dibujar_caja(frame, box_ojo_izq, (255, 0, 0), "Ojo izq")
                dibujar_caja(frame, box_ojo_der, (255, 0, 0), "Ojo der")

                # 4. Segmentación pupilar y PIR (A3)
                pir_valores = []
                if roi_ojo_izq is not None and roi_ojo_izq.size > 0 and box_ojo_izq is not None:
                    _, prep_izq = preprocess_eye_roi(roi_ojo_izq, scale_factor=SCALE_FACTOR)
                    diam_izq = calcular_diametro_iris(iris_izq_pts)
                    centro_izq = ((iris_izq_pts[0][0] - box_ojo_izq[0]) * SCALE_FACTOR, (iris_izq_pts[0][1] - box_ojo_izq[1]) * SCALE_FACTOR) if iris_izq_pts else None
                    elipse_izq, _, res_izq = segment_pupil(prep_izq, iris_center_roi=centro_izq, iris_radius_roi=(diam_izq / 2.0) * SCALE_FACTOR)
                    pir_izq = calcular_pir(elipse_izq, diam_izq, scale_factor=SCALE_FACTOR)
                    if pir_izq is not None:
                        pir_valores.append(pir_izq)
                    if mostrar_rois and res_izq is not None:
                        cv2.imshow("A3 - Pupila Ojo Izquierdo", res_izq)

                if roi_ojo_der is not None and roi_ojo_der.size > 0 and box_ojo_der is not None:
                    _, prep_der = preprocess_eye_roi(roi_ojo_der, scale_factor=SCALE_FACTOR)
                    diam_der = calcular_diametro_iris(iris_der_pts)
                    centro_der = ((iris_der_pts[0][0] - box_ojo_der[0]) * SCALE_FACTOR, (iris_der_pts[0][1] - box_ojo_der[1]) * SCALE_FACTOR) if iris_der_pts else None
                    elipse_der, _, res_der = segment_pupil(prep_der, iris_center_roi=centro_der, iris_radius_roi=(diam_der / 2.0) * SCALE_FACTOR)
                    pir_der = calcular_pir(elipse_der, diam_der, scale_factor=SCALE_FACTOR)
                    if pir_der is not None:
                        pir_valores.append(pir_der)
                    if mostrar_rois and res_der is not None:
                        cv2.imshow("A3 - Pupila Ojo Derecho", res_der)

                if pir_valores:
                    pir_prom = sum(pir_valores) / len(pir_valores)
                    pir_suavizado = pir_prom if pir_suavizado is None else (0.8 * pir_suavizado + 0.2 * pir_prom)

                pitch_eval = pitch if (pitch is not None and pitch > UMBRAL_PITCH_CABECEO) else 0.0
                yaw_eval = yaw if (yaw is not None and abs(yaw) > UMBRAL_YAW_DISTRACCION) else 0.0

                estado_global, score_f, score_s, alerta_critica, color_global = clasificador_a6.clasificar(
                    ear_promedio, perclos_actual, duracion_cierre, mar_actual,
                    bostezo_activo, pitch_eval, yaw_eval, pir_suavizado, sacadico_actual
                )

                if alerta_critica:
                    cv2.rectangle(frame, (0, 0), (ancho - 1, alto - 1), (0, 0, 255), 6)
                    if SONIDO_DISPONIBLE and (time.time() - ultimo_beep > 1.5):
                        winsound.Beep(1200, 250)
                        ultimo_beep = time.time()

            # FPS
            t_fps = time.time()
            dt = t_fps - tiempo_anterior
            tiempo_anterior = t_fps
            if dt > 0:
                fps_actual = 1.0 / dt
                fps_suavizado = fps_actual if fps_suavizado == 0 else (0.9 * fps_suavizado + 0.1 * fps_actual)

            # PANEL LATERAL DERECHO
            ancho_panel = 390
            panel = np.full((alto, ancho_panel, 3), (25, 25, 25), dtype=np.uint8)
            cv2.line(panel, (0, 0), (0, alto), (80, 80, 80), 2)

            cv2.putText(panel, "TELEMETRIA EN CABINA (TEL-328)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2)
            cv2.line(panel, (15, 42), (ancho_panel - 15, 42), (70, 70, 70), 1)

            cv2.putText(panel, "ESTADO ACTUAL:", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
            cv2.putText(panel, estado_global, (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_global, 2)

            pir_txt = f"{pir_suavizado:.2f}" if pir_suavizado is not None else "--"
            p_txt = f"{pitch:+.1f}" if pitch is not None else "--"
            y_txt = f"{yaw:+.1f}" if yaw is not None else "--"
            r_txt = f"{roll:+.1f}" if roll is not None else "--"

            lineas_info = [
                (f"EAR Ocular: {ear_promedio:.2f} (Cierre: {duracion_cierre:.1f}s)", (255, 255, 255)),
                (f"PERCLOS (90f): {perclos_actual:.1f}%", (255, 255, 255)),
                (f"MAR Boca: {mar_actual:.2f} {'[BOSTEZO]' if bostezo_activo else ''}", (255, 255, 255)),
                (f"PIR Pupila: {pir_txt} (Basal: {clasificador_a6.pir_basal:.2f})", (255, 255, 255)),
                (f"Inestabilidad Sacadica: {sacadico_actual:.1f}", (255, 255, 255)),
                (f"Pose 3D -> Pitch:{p_txt} Yaw:{y_txt} Roll:{r_txt}", (255, 255, 0)),
                (f"Estado Pose: {estado_pose}", (255, 255, 0)),
                (f"Riesgo Fatiga: {score_f:.0f}% | Ocular: {score_s:.0f}%", (180, 255, 180)),
            ]

            # Líneas de A4: estado/alertas y resumen de bostezos y mirada
            ind_a4 = estado_a4.indicadores or {}
            fmt_a4 = lambda v, f: "--" if v is None or v != v else f.format(v)  # v != v -> NaN
            if estado_a4.calibrando_s > 0:
                txt_a4, col_a4 = f"A4: calibrando, boca cerrada {estado_a4.calibrando_s:.0f}s", (255, 255, 0)
            else:
                txt_a4 = "A4: " + (" | ".join(estado_a4.alertas) if estado_a4.alertas else "NORMAL")
                col_a4 = [(0, 255, 0), (0, 165, 255), (0, 0, 255)][estado_a4.nivel]
            lineas_info += [
                (txt_a4, col_a4),
                (f"Bostezos 15min: {ind_a4.get('bostezos_15min', 0)} | Fij {fmt_a4(ind_a4.get('fijacion_actual_s'), '{:.1f}')}s"
                 f" | Sac {fmt_a4(ind_a4.get('sacadas_por_s'), '{:.2f}')}/s", (255, 255, 255)),
            ]

            y_pos = 135
            for texto, col in lineas_info:
                cv2.putText(panel, texto, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.47, col, 1)
                y_pos += 24

            cv2.line(panel, (15, alto - 95), (ancho_panel - 15, alto - 95), (70, 70, 70), 1)
            cv2.putText(panel, f"FPS: {fps_suavizado:.1f} | Pruebas A7: {contador_evidencias}/4", (15, alto - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 200), 1)
            cv2.putText(panel, "[S] Guardar Prueba  [V] ROIs  [Q] Salir", (15, alto - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            vista_combinada = np.hstack((frame, panel))
            cv2.imshow("Sistema Integrado de Monitoreo en Cabina - TEL-328", vista_combinada)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            elif tecla == ord("v"):
                mostrar_rois = not mostrar_rois
                if not mostrar_rois:
                    cv2.destroyWindow("A3 - Pupila Ojo Izquierdo")
                    cv2.destroyWindow("A3 - Pupila Ojo Derecho")
            elif tecla == ord("s"):
                contador_evidencias += 1
                nombre = carpeta_evidencias / f"a7_prueba{contador_evidencias}_{int(time.time())}.jpg"
                cv2.imwrite(str(nombre), vista_combinada)
                registrar_prueba_a7(
                    carpeta_evidencias, contador_evidencias, nombre.name,
                    estado_global, ear_promedio, perclos_actual, mar_actual,
                    pir_suavizado, pitch, yaw
                )
                print(f"[OK] Evidencia #{contador_evidencias} guardada en {nombre.name} ({estado_global})")

    cap.release()
    cv2.destroyAllWindows()

    resumen_a4 = monitor_a4.finish()
    print(f"[A4] Bostezos: {len(resumen_a4['bostezos'])} | Ventanas con alerta: "
          f"{resumen_a4['ventanas_con_alerta']}/{resumen_a4['ventanas']}")
    if args.salida:
        print(f"[OK] Indicadores A4 guardados en: {monitor_a4.save(args.salida, resumen_a4)}")


if __name__ == "__main__":
    main()
