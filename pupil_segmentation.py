import cv2
import numpy as np


def preprocess_eye_roi(eye_roi, scale_factor=3.0, clip_limit=2.0, tile_grid_size=(8, 8)):
    if eye_roi is None or eye_roi.size == 0:
        return None, None

    if len(eye_roi.shape) == 3:
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = eye_roi.copy()

    h, w = gray.shape[:2]
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    scaled_gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(scaled_gray)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    return scaled_gray, blurred


def segment_pupil(blurred_roi, iris_center_roi=None, iris_radius_roi=None, threshold_percentile=12):

    if blurred_roi is None:
        return None, None, None

    roi_to_thresh = blurred_roi.copy()
    mask_iris = np.ones_like(blurred_roi, dtype=np.uint8) * 255

    if iris_center_roi is not None and iris_radius_roi is not None and iris_radius_roi > 0:
        mask_iris = np.zeros_like(blurred_roi, dtype=np.uint8)
        cv2.circle(
            mask_iris,
            (int(iris_center_roi[0]), int(iris_center_roi[1])),
            int(iris_radius_roi * 0.85),
            255,
            -1
        )
        iris_pixels = blurred_roi[mask_iris == 255]
        if iris_pixels.size > 0:
            thresh_value = np.percentile(iris_pixels, threshold_percentile)
        else:
            thresh_value = np.percentile(blurred_roi, threshold_percentile)
    else:
        thresh_value = np.percentile(blurred_roi, threshold_percentile)

    _, thresh = cv2.threshold(roi_to_thresh, thresh_value, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.bitwise_and(thresh, mask_iris)

    # Operaciones morfológicas para limpiar imperfecciones
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pupil_ellipse = None
    masked_pupil = cv2.cvtColor(blurred_roi, cv2.COLOR_GRAY2BGR)

    if contours:
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if len(c) >= 5 and area > 30:
                _, (w_e, h_e), _ = cv2.fitEllipse(c)
                if w_e > 0 and h_e > 0:
                    aspect_ratio = max(w_e, h_e) / min(w_e, h_e)
                    if aspect_ratio < 1.8:
                        valid_contours.append(c)

        if valid_contours:
            best_contour = max(valid_contours, key=cv2.contourArea)
            pupil_ellipse = cv2.fitEllipse(best_contour)

            cv2.ellipse(masked_pupil, pupil_ellipse, (0, 255, 0), 2)
            cv2.circle(
                masked_pupil,
                (int(pupil_ellipse[0][0]), int(pupil_ellipse[0][1])),
                2,
                (0, 0, 255),
                -1
            )

    return pupil_ellipse, thresh, masked_pupil


def calcular_ear(puntos_ear):
    if len(puntos_ear) != 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = [np.array(p, dtype=np.float32) for p in puntos_ear]

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0:
        return 0.0

    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def calcular_diametro_iris(puntos_iris):
    if len(puntos_iris) < 5:
        return 0.0

    p_izq = np.array(puntos_iris[1], dtype=np.float32)
    p_der = np.array(puntos_iris[3], dtype=np.float32)
    p_arr = np.array(puntos_iris[2], dtype=np.float32)
    p_aba = np.array(puntos_iris[4], dtype=np.float32)

    d_horiz = np.linalg.norm(p_izq - p_der)
    d_vert = np.linalg.norm(p_arr - p_aba)

    return float((d_horiz + d_vert) / 2.0)


def calcular_pir(pupil_ellipse, iris_diameter_px, scale_factor=3.0):
    if pupil_ellipse is None or iris_diameter_px <= 0:
        return None

    (_, _), (axes_a, axes_b), _ = pupil_ellipse
    pupil_diameter_scaled = (axes_a + axes_b) / 2.0
    pupil_diameter_px = pupil_diameter_scaled / scale_factor

    pir = pupil_diameter_px / iris_diameter_px
    return float(np.clip(pir, 0.0, 1.0))
