import argparse
import csv
import cv2
import mediapipe as mp
import time
from pathlib import Path

from pupil_segmentation import (
    preprocess_eye_roi,
    segment_pupil,
    calcular_ear,
    calcular_diametro_iris,
    calcular_pir
)
from head_pose import (
    PITCH_UMBRAL_CABECEO,
    YAW_UMBRAL_DISTRACCION,
    obtener_puntos_pose_2d,
    estimar_pose_cabeza,
    dibujar_vector_pose,
    evaluar_estado_pose
)
from a4 import A4Monitor, landmarks_to_array, blendshapes_to_dict, pose_from_result


# CONFIGURACIÓN
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (BASE_DIR / "models" / "face_landmarker.task").resolve()

EAR_UMBRAL = 0.21
TIEMPO_MICROSUEÑO_SEG = 3.0
SCALE_FACTOR = 3.0


# LANDMARKS DE REGIONES FACIALES
LEFT_EYE = [
    263, 249, 390, 373, 374, 380, 381, 382,
    362, 398, 384, 385, 386, 387, 388, 466
]

RIGHT_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]

# 6 puntos específicos para EAR: [p1, p2, p3, p4, p5, p6]
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]

LEFT_IRIS = [473, 474, 475, 476, 477]
RIGHT_IRIS = [468, 469, 470, 471, 472]

MOUTH = [
    61, 146, 91, 181, 84, 17,
    314, 405, 321, 375, 291,
    409, 270, 269, 267, 0,
    37, 39, 40, 185
]


# FUNCIONES AUXILIARES
def obtener_puntos(landmarks, indices, ancho, alto):
    puntos = []
    for indice in indices:
        if indice >= len(landmarks):
            continue
        landmark = landmarks[indice]
        x = int(landmark.x * ancho)
        y = int(landmark.y * alto)
        puntos.append((x, y))
    return puntos


def dibujar_puntos(frame, puntos, color, radio=2):
    for punto in puntos:
        cv2.circle(frame, punto, radio, color, -1)


def dibujar_contorno(frame, puntos, color):
    if len(puntos) < 2:
        return
    for i in range(len(puntos) - 1):
        cv2.line(frame, puntos[i], puntos[i + 1], color, 1)
    cv2.line(frame, puntos[-1], puntos[0], color, 1)


def obtener_roi(frame_limpio, puntos, margen=10):
    if not puntos:
        return None, None
    xs = [p[0] for p in puntos]
    ys = [p[1] for p in puntos]

    x1 = max(min(xs) - margen, 0)
    y1 = max(min(ys) - margen, 0)
    x2 = min(max(xs) + margen, frame_limpio.shape[1])
    y2 = min(max(ys) + margen, frame_limpio.shape[0])

    if x2 <= x1 or y2 <= y1:
        return None, None

    roi = frame_limpio[y1:y2, x1:x2].copy()
    return roi, (x1, y1, x2, y2)


def dibujar_caja(frame, caja, color, texto):
    if caja is None:
        return
    x1, y1, x2, y2 = caja
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    cv2.putText(
        frame, texto, (x1, max(y1 - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
    )


def registrar_evidencia_a7(carpeta_evidencias, num_prueba, nombre_img, pitch, yaw, roll, ear, pir, estado_pose, estado_ocular):
    archivo_csv = carpeta_evidencias / "registro_pruebas_a7.csv"
    existe = archivo_csv.exists()
    with open(archivo_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow([
                "Prueba",
                "Timestamp",
                "Pitch_deg",
                "Yaw_deg",
                "Roll_deg",
                "EAR",
                "PIR",
                "Estado_Pose",
                "Estado_Ocular",
                "Imagen"
            ])
        writer.writerow([
            num_prueba,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            f"{pitch:.2f}" if pitch is not None else "",
            f"{yaw:.2f}" if yaw is not None else "",
            f"{roll:.2f}" if roll is not None else "",
            f"{ear:.2f}",
            f"{pir:.2f}" if pir is not None else "",
            estado_pose,
            estado_ocular,
            nombre_img.name
        ])


# PROGRAMA PRINCIPAL
def main():
    parser = argparse.ArgumentParser(description="Análisis ocular, dinámica pupilar y pose 3D de cabeza")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Ruta al archivo de video (ej. UTA-RLDD). Si se omite, usa la webcam."
    )
    parser.add_argument(
        "--cam",
        type=int,
        default=0,
        help="Índice de la cámara cuando no se usa --video (0 por defecto; prueba 1, 2... si abre otra cámara)."
    )
    parser.add_argument(
        "--calib-a4",
        type=float,
        default=5.0,
        help="Segundos iniciales con la boca cerrada para calibrar el umbral de bostezo (A4)."
    )
    parser.add_argument(
        "--salida",
        type=str,
        default=None,
        help="Carpeta donde guardar los indicadores de A4 (indicadores_a4.csv, resumen_a4.json)."
    )
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
        output_facial_transformation_matrixes=True   # A4: pose de la cabeza
    )

    print("[OK] Fuente iniciada correctamente.")
    print("[OK] MediaPipe Face Landmarker + Modulo Head Pose 3D iniciados.")
    print(f"[INFO] Umbrales A7 -> Cabeceo: Pitch > {PITCH_UMBRAL_CABECEO:.0f} deg | Distraccion: |Yaw| > {YAW_UMBRAL_DISTRACCION:.0f} deg")
    print("Controles: Q = cerrar | S = guardar captura y registrar prueba de validacion")

    tiempo_anterior = time.time()
    fps_suavizado = 0.0
    ultimo_timestamp = 0
    pir_suavizado = None
    contador_evidencias = 0

    # Variables de estado A3 (Fatiga ocular)
    inicio_cierre_ojos = None
    duracion_cierre = 0.0
    contador_parpadeos = 0
    ojo_estaba_cerrado = False

    # Estado A4 (Bostezos y mirada)
    monitor_a4 = A4Monitor(calib_s=args.calib_a4, history_s=None if es_video else 120.0)

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Fin del video o error de captura.")
                break

            if not es_video:
                frame = cv2.flip(frame, 1)

            frame_limpio = frame.copy()
            alto, ancho = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame_limpio, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            if es_video:
                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                tiempo_actual_seg = timestamp_ms / 1000.0
            else:
                timestamp_ms = int(time.monotonic() * 1000)
                tiempo_actual_seg = time.time()

            if timestamp_ms <= ultimo_timestamp:
                timestamp_ms = ultimo_timestamp + 1
            ultimo_timestamp = timestamp_ms

            resultado = landmarker.detect_for_video(mp_image, timestamp_ms)

            # PROCESAMIENTO A4: BOSTEZOS Y MIRADA
            pts_a4 = (landmarks_to_array(resultado.face_landmarks[0], ancho, alto)
                      if resultado.face_landmarks else None)
            estado_a4 = monitor_a4.update(
                pts_a4, tiempo_actual_seg,
                pose=pose_from_result(resultado),
                blendshapes=blendshapes_to_dict(resultado)
            )

            roi_ojo_izq = None
            roi_ojo_der = None
            box_ojo_izq = None
            box_ojo_der = None
            iris_izq_pts = []
            iris_der_pts = []

            ear_promedio = 0.0
            estado_alerta = "NORMAL"
            color_estado = (0, 255, 0)

            pitch, yaw, roll = None, None, None
            estado_pose = "SIN ROSTRO"
            color_pose = (0, 0, 255)

            if resultado.face_landmarks:
                landmarks = resultado.face_landmarks[0]

                ojo_izquierdo = obtener_puntos(landmarks, LEFT_EYE, ancho, alto)
                ojo_derecho = obtener_puntos(landmarks, RIGHT_EYE, ancho, alto)
                iris_izq_pts = obtener_puntos(landmarks, LEFT_IRIS, ancho, alto)
                iris_der_pts = obtener_puntos(landmarks, RIGHT_IRIS, ancho, alto)
                boca = obtener_puntos(landmarks, MOUTH, ancho, alto)

                # Cálculo de EAR
                ear_izq_pts = obtener_puntos(landmarks, LEFT_EYE_EAR, ancho, alto)
                ear_der_pts = obtener_puntos(landmarks, RIGHT_EYE_EAR, ancho, alto)
                ear_izq = calcular_ear(ear_izq_pts)
                ear_der = calcular_ear(ear_der_pts)
                ear_promedio = (ear_izq + ear_der) / 2.0

                # Lógica de parpadeo prolongado (3.0 s)
                if ear_promedio < EAR_UMBRAL:
                    if not ojo_estaba_cerrado:
                        ojo_estaba_cerrado = True
                        inicio_cierre_ojos = tiempo_actual_seg
                    duracion_cierre = tiempo_actual_seg - inicio_cierre_ojos

                    if duracion_cierre >= TIEMPO_MICROSUEÑO_SEG:
                        estado_alerta = "ALERTA: MICROSUENO"
                        color_estado = (0, 0, 255)
                    else:
                        estado_alerta = "OJO CERRADO"
                        color_estado = (0, 165, 255)
                else:
                    if ojo_estaba_cerrado:
                        contador_parpadeos += 1
                    ojo_estaba_cerrado = False
                    inicio_cierre_ojos = None
                    duracion_cierre = 0.0

                # PROCESAMIENTO A7: POSE E INCLINACIÓN DE LA CABEZA 3D
                puntos_pose_2d = obtener_puntos_pose_2d(landmarks, ancho, alto)
                angulos_pose, vector_nariz, _ = estimar_pose_cabeza(puntos_pose_2d, ancho, alto)
                if angulos_pose is not None and vector_nariz is not None:
                    pitch, yaw, roll = angulos_pose
                    estado_pose, color_pose = evaluar_estado_pose(pitch, yaw, roll)
                    dibujar_vector_pose(
                        frame,
                        vector_nariz[0],
                        vector_nariz[1],
                        puntos_2d=puntos_pose_2d
                    )

                # Dibujar regiones faciales
                dibujar_contorno(frame, ojo_izquierdo, (255, 0, 0))
                dibujar_contorno(frame, ojo_derecho, (255, 0, 0))
                dibujar_puntos(frame, iris_izq_pts, (0, 255, 255), 2)
                dibujar_puntos(frame, iris_der_pts, (0, 255, 255), 2)
                dibujar_contorno(frame, boca, (255, 0, 255))

                # Extraer ROI limpios
                roi_ojo_izq, box_ojo_izq = obtener_roi(frame_limpio, ojo_izquierdo, margen=15)
                roi_ojo_der, box_ojo_der = obtener_roi(frame_limpio, ojo_derecho, margen=15)

                dibujar_caja(frame, box_ojo_izq, (255, 0, 0), "Ojo izq.")
                dibujar_caja(frame, box_ojo_der, (255, 0, 0), "Ojo der.")

            else:
                estado_alerta = "ROSTRO NO DETECTADO"
                color_estado = (0, 0, 255)

            # PROCESAMIENTO A3: SEGMENTACIÓN PUPILAR Y PIR
            pir_valores = []

            if roi_ojo_izq is not None and roi_ojo_izq.size > 0 and box_ojo_izq is not None:
                _, preprocessed_izq = preprocess_eye_roi(roi_ojo_izq, scale_factor=SCALE_FACTOR)
                diam_iris_izq = calcular_diametro_iris(iris_izq_pts)
                centro_iris_izq_roi = None
                if iris_izq_pts:
                    cx = (iris_izq_pts[0][0] - box_ojo_izq[0]) * SCALE_FACTOR
                    cy = (iris_izq_pts[0][1] - box_ojo_izq[1]) * SCALE_FACTOR
                    centro_iris_izq_roi = (cx, cy)

                elipse_izq, _, res_pupila_izq = segment_pupil(
                    preprocessed_izq,
                    iris_center_roi=centro_iris_izq_roi,
                    iris_radius_roi=(diam_iris_izq / 2.0) * SCALE_FACTOR
                )
                pir_izq = calcular_pir(elipse_izq, diam_iris_izq, scale_factor=SCALE_FACTOR)
                if pir_izq is not None:
                    pir_valores.append(pir_izq)
                if res_pupila_izq is not None:
                    cv2.imshow("A3 - Pupila Ojo Izquierdo", res_pupila_izq)

            if roi_ojo_der is not None and roi_ojo_der.size > 0 and box_ojo_der is not None:
                _, preprocessed_der = preprocess_eye_roi(roi_ojo_der, scale_factor=SCALE_FACTOR)
                diam_iris_der = calcular_diametro_iris(iris_der_pts)
                centro_iris_der_roi = None
                if iris_der_pts:
                    cx = (iris_der_pts[0][0] - box_ojo_der[0]) * SCALE_FACTOR
                    cy = (iris_der_pts[0][1] - box_ojo_der[1]) * SCALE_FACTOR
                    centro_iris_der_roi = (cx, cy)

                elipse_der, _, res_pupila_der = segment_pupil(
                    preprocessed_der,
                    iris_center_roi=centro_iris_der_roi,
                    iris_radius_roi=(diam_iris_der / 2.0) * SCALE_FACTOR
                )
                pir_der = calcular_pir(elipse_der, diam_iris_der, scale_factor=SCALE_FACTOR)
                if pir_der is not None:
                    pir_valores.append(pir_der)
                if res_pupila_der is not None:
                    cv2.imshow("A3 - Pupila Ojo Derecho", res_pupila_der)

            if pir_valores:
                pir_promedio = sum(pir_valores) / len(pir_valores)
                if pir_suavizado is None:
                    pir_suavizado = pir_promedio
                else:
                    pir_suavizado = 0.8 * pir_suavizado + 0.2 * pir_promedio

            # FPS
            tiempo_fps = time.time()
            diferencia = tiempo_fps - tiempo_anterior
            tiempo_anterior = tiempo_fps
            if diferencia > 0:
                fps_actual = 1.0 / diferencia
                fps_suavizado = fps_actual if fps_suavizado == 0 else (0.9 * fps_suavizado + 0.1 * fps_actual)

            # MÉTRICAS EN VENTANA PRINCIPAL
            cv2.putText(frame, f"Ocular: {estado_alerta}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_estado, 2)
            cv2.putText(frame, f"Pose 3D: {estado_pose}", (15, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_pose, 2)

            if pitch is not None and yaw is not None and roll is not None:
                texto_pose = f"Pitch: {pitch:+.1f} deg | Yaw: {yaw:+.1f} deg | Roll: {roll:+.1f} deg"
            else:
                texto_pose = "Pitch: -- | Yaw: -- | Roll: --"
            cv2.putText(frame, texto_pose, (15, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 0), 2)

            cv2.putText(frame, f"EAR: {ear_promedio:.2f} (Cierre: {duracion_cierre:.1f}s)", (15, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            pir_texto = f"{pir_suavizado:.2f}" if pir_suavizado is not None else "--"
            cv2.putText(frame, f"PIR: {pir_texto} | Parpadeos: {contador_parpadeos}", (15, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps_suavizado:.1f} | Res: {ancho}x{alto} | Pruebas A7: {contador_evidencias}/4", (15, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            monitor_a4.draw(frame, estado_a4, pts_a4, origin=(15, 196))

            cv2.imshow("Monitoreo Ocular y Pose 3D de Cabeza", frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == ord("s"):
                contador_evidencias += 1
                carpeta_evidencias = BASE_DIR / "evidencias"
                carpeta_evidencias.mkdir(exist_ok=True)
                nombre = carpeta_evidencias / f"a7_prueba{contador_evidencias}_{int(time.time())}.jpg"
                cv2.imwrite(str(nombre), frame)
                registrar_evidencia_a7(
                    carpeta_evidencias,
                    contador_evidencias,
                    nombre,
                    pitch,
                    yaw,
                    roll,
                    ear_promedio,
                    pir_suavizado,
                    estado_pose,
                    estado_alerta
                )
                p_str = f"{pitch:+.1f}°" if pitch is not None else "--"
                y_str = f"{yaw:+.1f}°" if yaw is not None else "--"
                r_str = f"{roll:+.1f}°" if roll is not None else "--"
                print(
                    f"[OK] Prueba #{contador_evidencias} guardada: {nombre.name} | "
                    f"Pitch={p_str}, Yaw={y_str}, Roll={r_str} | Estado={estado_pose}"
                )

    cap.release()
    cv2.destroyAllWindows()

    resumen_a4 = monitor_a4.finish()
    print(f"[A4] Bostezos: {len(resumen_a4['bostezos'])} | Ventanas con alerta: "
          f"{resumen_a4['ventanas_con_alerta']}/{resumen_a4['ventanas']}")
    if args.salida:
        carpeta = monitor_a4.save(args.salida, resumen_a4)
        print(f"[OK] Indicadores A4 guardados en: {carpeta}")
    print("[OK] Ejecución finalizada correctamente.")


if __name__ == "__main__":
    main()
