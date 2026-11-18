import cv2
import numpy as np


# Índices de los 6 puntos clave 2D en MediaPipe Face Landmarker:
# 1: Punta de la nariz
# 152: Mentón
# 33: Canto externo del ojo derecho
# 263: Canto externo del ojo izquierdo
# 61: Comisura derecha de la boca
# 291: Comisura izquierda de la boca
POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]

# Modelo geométrico 3D canónico del rostro (en milímetros)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Punta de la nariz (1)
    (0.0, -330.0, -65.0),        # Mentón (152)
    (-225.0, 170.0, -135.0),     # Canto externo ojo derecho (33)
    (225.0, 170.0, -135.0),      # Canto externo ojo izquierdo (263)
    (-150.0, -150.0, -125.0),    # Comisura derecha de la boca (61)
    (150.0, -150.0, -125.0)      # Comisura izquierda de la boca (291)
], dtype=np.float64)

# Umbrales de validación A7 (en grados)
PITCH_UMBRAL_CABECEO = 15.0
YAW_UMBRAL_DISTRACCION = 28.0
ROLL_UMBRAL_INCLINACION = 20.0

# Color celeste en formato BGR para el vector de dirección 3D
COLOR_VECTOR_CELESTE = (255, 255, 0)


def obtener_puntos_pose_2d(landmarks, ancho, alto):
    """Extrae las coordenadas 2D (x, y) de los 6 puntos clave del rostro."""
    puntos_2d = []
    for idx in POSE_LANDMARKS:
        if idx >= len(landmarks):
            return None
        lm = landmarks[idx]
        x = lm.x * ancho
        y = lm.y * alto
        puntos_2d.append((x, y))
    return np.array(puntos_2d, dtype=np.float64)


def estimar_pose_cabeza(puntos_2d, ancho, alto, longitud_vector=500.0):
    """
    Empareja los 6 puntos 2D del rostro con el modelo 3D canónico mediante cv2.solvePnP,
    calcula la matriz de rotación con cv2.Rodrigues y obtiene los ángulos:
      - Pitch (cabeceo: positivo al inclinar la cabeza hacia abajo)
      - Yaw (giro lateral por distracción: magnitud > 28° al girar hacia los lados)
      - Roll (inclinación lateral de la cabeza)
    Además proyecta el vector 3D de dirección sobre la punta de la nariz.
    """
    if puntos_2d is None or len(puntos_2d) != 6:
        return None, None, None

    focal_length = float(ancho)
    center = (ancho / 2.0, alto / 2.0)
    camera_matrix = np.array([
        [focal_length, 0.0, center[0]],
        [0.0, focal_length, center[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    exito, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D,
        puntos_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not exito:
        return None, None, None

    # Matriz de rotación 3x3 a partir del vector de rotación (Rodrigues)
    rmat, _ = cv2.Rodrigues(rvec)

    # Alinear ejes del modelo canónico (Y arriba, Z hacia cámara) con cámara OpenCV (Y abajo, Z hacia escena)
    rmat_alineada = rmat @ np.diag([1.0, -1.0, -1.0])
    angulos, _, _, _, _, _ = cv2.RQDecomp3x3(rmat_alineada)

    pitch = float(angulos[0])   # Cabeceo: > +15° al inclinar hacia abajo
    yaw = float(-angulos[1])    # Giro lateral: |Yaw| > 28° al girar hacia un lado
    roll = float(angulos[2])    # Inclinación lateral

    # Proyección del vector 3D desde la punta de la nariz (0, 0, longitud_vector)
    extremo_3d = np.array([(0.0, 0.0, float(longitud_vector))], dtype=np.float64)
    punto_proyectado_2d, _ = cv2.projectPoints(
        extremo_3d, rvec, tvec, camera_matrix, dist_coeffs
    )

    p1 = (int(puntos_2d[0][0]), int(puntos_2d[0][1]))
    p2 = (int(punto_proyectado_2d[0][0][0]), int(punto_proyectado_2d[0][0][1]))

    return (pitch, yaw, roll), (p1, p2), (rvec, tvec)


def dibujar_vector_pose(frame, punto_nariz, punto_proyectado, puntos_2d=None, color=COLOR_VECTOR_CELESTE):
    """Dibuja los 6 puntos clave 2D y el vector celeste de dirección 3D proyectado en la nariz."""
    if puntos_2d is not None:
        for pt in puntos_2d:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

    if punto_nariz is not None and punto_proyectado is not None:
        cv2.arrowedLine(
            frame,
            punto_nariz,
            punto_proyectado,
            color,
            2,
            tipLength=0.15
        )
        cv2.circle(frame, punto_nariz, 4, (0, 0, 255), -1)


def evaluar_estado_pose(
    pitch,
    yaw,
    roll,
    pitch_umbral=PITCH_UMBRAL_CABECEO,
    yaw_umbral=YAW_UMBRAL_DISTRACCION,
    roll_umbral=ROLL_UMBRAL_INCLINACION
):
    """Evalúa alertas de cabeceo (Pitch > 15°) y distracción por giro lateral (|Yaw| > 28°)."""
    if pitch > pitch_umbral and abs(yaw) > yaw_umbral:
        return "ALERTA: CABECEO Y DISTRACCION", (0, 0, 255)
    if pitch > pitch_umbral:
        return "ALERTA: CABECEO (PITCH > 15)", (0, 0, 255)
    if abs(yaw) > yaw_umbral:
        return "ALERTA: DISTRACCION (YAW > 28)", (0, 165, 255)
    if abs(roll) > roll_umbral:
        return "INCLINACION LATERAL (ROLL)", (0, 255, 255)
    return "POSE NORMAL", (0, 255, 0)
