import cv2
import mediapipe as mp
import time
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"


# ============================================================
# LANDMARKS DE REGIONES FACIALES
# ============================================================

# Ojo izquierdo anatómico de la persona
LEFT_EYE = [
    263, 249, 390, 373, 374, 380, 381, 382,
    362, 398, 384, 385, 386, 387, 388, 466
]

# Ojo derecho anatómico de la persona
RIGHT_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]

# Iris izquierdo
LEFT_IRIS = [
    473, 474, 475, 476, 477
]

# Iris derecho
RIGHT_IRIS = [
    468, 469, 470, 471, 472
]

# Contorno exterior de labios
MOUTH = [
    61, 146, 91, 181, 84, 17,
    314, 405, 321, 375, 291,
    409, 270, 269, 267, 0,
    37, 39, 40, 185
]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def obtener_puntos(landmarks, indices, ancho, alto):
    """
    Convierte landmarks normalizados de MediaPipe
    a coordenadas reales en píxeles.
    """

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
    """
    Dibuja una lista de puntos sobre el frame.
    """

    for punto in puntos:
        cv2.circle(
            frame,
            punto,
            radio,
            color,
            -1
        )


def dibujar_contorno(frame, puntos, color):
    """
    Une visualmente los puntos de una región facial.
    """

    if len(puntos) < 2:
        return

    for i in range(len(puntos) - 1):
        cv2.line(
            frame,
            puntos[i],
            puntos[i + 1],
            color,
            1
        )

    # Cerrar el contorno
    cv2.line(
        frame,
        puntos[-1],
        puntos[0],
        color,
        1
    )


def obtener_roi(frame_limpio, puntos, margen=10):
    """
    Obtiene un recorte rectangular alrededor
    de una región facial.
    """

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
    """
    Dibuja un bounding box con una etiqueta.
    """

    if caja is None:
        return

    x1, y1, x2, y2 = caja

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        1
    )

    cv2.putText(
        frame,
        texto,
        (x1, max(y1 - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1
    )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    # --------------------------------------------------------
    # Verificar modelo
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        print("[ERROR] No se encontró el modelo:")
        print(MODEL_PATH)

        print(
            "\nDescarga face_landmarker.task "
            "dentro de la carpeta models/."
        )

        return


    # --------------------------------------------------------
    # Inicializar cámara
    # --------------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return


    # --------------------------------------------------------
    # Configurar MediaPipe Tasks
    # --------------------------------------------------------

    BaseOptions = mp.tasks.BaseOptions

    FaceLandmarker = (
        mp.tasks.vision.FaceLandmarker
    )

    FaceLandmarkerOptions = (
        mp.tasks.vision.FaceLandmarkerOptions
    )

    RunningMode = (
        mp.tasks.vision.RunningMode
    )


    options = FaceLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),

        running_mode=RunningMode.VIDEO,

        num_faces=1,

        min_face_detection_confidence=0.5,

        min_face_presence_confidence=0.5,

        min_tracking_confidence=0.5
    )


    print("[OK] Cámara iniciada correctamente.")
    print("[OK] MediaPipe Face Landmarker iniciado.")
    print()
    print("Controles:")
    print("  Q = cerrar")
    print("  S = guardar captura")
    print()


    # --------------------------------------------------------
    # Variables FPS
    # --------------------------------------------------------

    tiempo_anterior = time.time()

    fps_suavizado = 0.0

    ultimo_timestamp = 0


    # --------------------------------------------------------
    # Crear Face Landmarker
    # --------------------------------------------------------

    with FaceLandmarker.create_from_options(options) as landmarker:

        while True:

            ret, frame = cap.read()

            if not ret:
                print("[ERROR] No se pudo obtener un frame.")
                break


            # ------------------------------------------------
            # Modo espejo
            # ------------------------------------------------

            frame = cv2.flip(frame, 1)


            # IMPORTANTE:
            # Guardamos una copia sin dibujos.
            # Los ROI se extraerán desde esta imagen.
            frame_limpio = frame.copy()


            alto, ancho = frame.shape[:2]


            # ------------------------------------------------
            # Convertir BGR -> RGB
            # ------------------------------------------------

            frame_rgb = cv2.cvtColor(
                frame_limpio,
                cv2.COLOR_BGR2RGB
            )


            # ------------------------------------------------
            # Crear imagen MediaPipe
            # ------------------------------------------------

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb
            )


            # ------------------------------------------------
            # Timestamp obligatorio en modo VIDEO
            # ------------------------------------------------

            timestamp_ms = int(
                time.monotonic() * 1000
            )

            if timestamp_ms <= ultimo_timestamp:
                timestamp_ms = ultimo_timestamp + 1

            ultimo_timestamp = timestamp_ms


            # ------------------------------------------------
            # Detectar rostro
            # ------------------------------------------------

            resultado = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )


            # Inicializamos ROI vacíos
            roi_ojo_izq = None
            roi_ojo_der = None
            roi_boca = None


            # =================================================
            # ROSTRO DETECTADO
            # =================================================

            if resultado.face_landmarks:

                cv2.putText(
                    frame,
                    "Rostro detectado",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )


                # Solo trabajamos con una persona
                landmarks = resultado.face_landmarks[0]


                # ---------------------------------------------
                # Dibujar TODOS los landmarks
                # ---------------------------------------------

                for landmark in landmarks:

                    x = int(
                        landmark.x * ancho
                    )

                    y = int(
                        landmark.y * alto
                    )

                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )


                # ---------------------------------------------
                # Obtener regiones faciales
                # ---------------------------------------------

                ojo_izquierdo = obtener_puntos(
                    landmarks,
                    LEFT_EYE,
                    ancho,
                    alto
                )

                ojo_derecho = obtener_puntos(
                    landmarks,
                    RIGHT_EYE,
                    ancho,
                    alto
                )

                iris_izquierdo = obtener_puntos(
                    landmarks,
                    LEFT_IRIS,
                    ancho,
                    alto
                )

                iris_derecho = obtener_puntos(
                    landmarks,
                    RIGHT_IRIS,
                    ancho,
                    alto
                )

                boca = obtener_puntos(
                    landmarks,
                    MOUTH,
                    ancho,
                    alto
                )


                # ---------------------------------------------
                # Dibujar regiones diferenciadas
                # ---------------------------------------------

                # Ojos
                dibujar_puntos(
                    frame,
                    ojo_izquierdo,
                    (255, 0, 0),
                    2
                )

                dibujar_puntos(
                    frame,
                    ojo_derecho,
                    (255, 0, 0),
                    2
                )


                # Iris
                dibujar_puntos(
                    frame,
                    iris_izquierdo,
                    (0, 255, 255),
                    3
                )

                dibujar_puntos(
                    frame,
                    iris_derecho,
                    (0, 255, 255),
                    3
                )


                # Boca
                dibujar_puntos(
                    frame,
                    boca,
                    (255, 0, 255),
                    2
                )


                # Dibujar contornos
                dibujar_contorno(
                    frame,
                    ojo_izquierdo,
                    (255, 0, 0)
                )

                dibujar_contorno(
                    frame,
                    ojo_derecho,
                    (255, 0, 0)
                )

                dibujar_contorno(
                    frame,
                    iris_izquierdo,
                    (0, 255, 255)
                )

                dibujar_contorno(
                    frame,
                    iris_derecho,
                    (0, 255, 255)
                )

                dibujar_contorno(
                    frame,
                    boca,
                    (255, 0, 255)
                )


                # ---------------------------------------------
                # Extraer ROI desde el FRAME LIMPIO
                # ---------------------------------------------

                roi_ojo_izq, box_ojo_izq = obtener_roi(
                    frame_limpio,
                    ojo_izquierdo,
                    margen=15
                )

                roi_ojo_der, box_ojo_der = obtener_roi(
                    frame_limpio,
                    ojo_derecho,
                    margen=15
                )

                roi_boca, box_boca = obtener_roi(
                    frame_limpio,
                    boca,
                    margen=15
                )


                # ---------------------------------------------
                # Dibujar bounding boxes
                # ---------------------------------------------

                dibujar_caja(
                    frame,
                    box_ojo_izq,
                    (255, 0, 0),
                    "Ojo izq."
                )

                dibujar_caja(
                    frame,
                    box_ojo_der,
                    (255, 0, 0),
                    "Ojo der."
                )

                dibujar_caja(
                    frame,
                    box_boca,
                    (255, 0, 255),
                    "Boca"
                )


                # ---------------------------------------------
                # Mostrar cantidad de landmarks
                # ---------------------------------------------

                cv2.putText(
                    frame,
                    f"Landmarks: {len(landmarks)}",
                    (15, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1
                )


            # =================================================
            # ROSTRO NO DETECTADO
            # =================================================

            else:

                cv2.putText(
                    frame,
                    "Rostro NO detectado",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )


            # =================================================
            # FPS
            # =================================================

            tiempo_actual = time.time()

            diferencia = (
                tiempo_actual - tiempo_anterior
            )

            tiempo_anterior = tiempo_actual


            if diferencia > 0:

                fps_actual = 1.0 / diferencia

                # Suavizado para que el valor
                # no cambie de forma tan brusca
                if fps_suavizado == 0:
                    fps_suavizado = fps_actual

                else:
                    fps_suavizado = (
                        0.9 * fps_suavizado
                        + 0.1 * fps_actual
                    )


            cv2.putText(
                frame,
                f"FPS: {fps_suavizado:.1f}",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Resolucion: {ancho}x{alto}",
                (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


            # =================================================
            # MOSTRAR VENTANA PRINCIPAL
            # =================================================

            cv2.imshow(
                "A2 - Landmarks faciales",
                frame
            )


            # =================================================
            # MOSTRAR ROI
            # =================================================

            if (
                roi_ojo_izq is not None
                and roi_ojo_izq.size > 0
            ):

                cv2.imshow(
                    "ROI - Ojo izquierdo",
                    roi_ojo_izq
                )


            if (
                roi_ojo_der is not None
                and roi_ojo_der.size > 0
            ):

                cv2.imshow(
                    "ROI - Ojo derecho",
                    roi_ojo_der
                )


            if (
                roi_boca is not None
                and roi_boca.size > 0
            ):

                cv2.imshow(
                    "ROI - Boca",
                    roi_boca
                )


            # =================================================
            # TECLADO
            # =================================================

            tecla = cv2.waitKey(1) & 0xFF


            # Q -> salir
            if tecla == ord("q"):
                break


            # S -> guardar evidencia
            if tecla == ord("s"):

                carpeta_evidencias = (
                    BASE_DIR / "evidencias"
                )

                carpeta_evidencias.mkdir(
                    exist_ok=True
                )

                nombre = (
                    carpeta_evidencias
                    / f"a2_{int(time.time())}.jpg"
                )

                cv2.imwrite(
                    str(nombre),
                    frame
                )

                print(
                    f"[OK] Evidencia guardada: {nombre}"
                )


    # ========================================================
    # CERRAR
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    print("[OK] Cámara cerrada correctamente.")


if __name__ == "__main__":
    main()
