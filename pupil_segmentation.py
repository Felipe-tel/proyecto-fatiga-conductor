import cv2
import numpy as np

def preprocess_eye_roi(eye_roi, scale_factor=3.0, clip_limit=2.0, tile_grid_size=(8, 8)):
    
    if eye_roi is None or eye_roi.size == 0:
        return None, None

    # 1. Escala de grises
    if len(eye_roi.shape) == 3:
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = eye_roi.copy()

    # 2. Escalamiento mediante interpolación bicúbica 
    h, w = gray.shape[:2]
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    scaled_gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 3. Ecualización adaptativa de histograma (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(scaled_gray)

    # 4. Reducción de ruido
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    return scaled_gray, blurred

def segment_pupil(blurred_roi, threshold_percentile=10):
    if blurred_roi is None:
        return None, None, None

    # Determinación de umbral dinámico basado en la región más oscura (pupila)
    thresh_value = np.percentile(blurred_roi, threshold_percentile)
    _, thresh = cv2.threshold(blurred_roi, thresh_value, 255, cv2.THRESH_BINARY_INV)

    # Operaciones morfológicas para limpiar imperfecciones
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Búsqueda de contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pupil_ellipse = None
    masked_pupil = cv2.cvtColor(blurred_roi, cv2.COLOR_GRAY2BGR)

    if contours:
        # Seleccionar el contorno con forma más circular y área relevante
        valid_contours = [c for c in contours if len(c) >= 5 and cv2.contourArea(c) > 50]
        
        if valid_contours:
            best_contour = max(valid_contours, key=cv2.contourArea)
            
            # Ajuste de elipse
            pupil_ellipse = cv2.fitEllipse(best_contour)
            
            cv2.ellipse(masked_pupil, pupil_ellipse, (0, 255, 0), 2)
            cv2.circle(masked_pupil, (int(pupil_ellipse[0][0]), int(pupil_ellipse[0][1])), 2, (0, 0, 255), -1)

    return pupil_ellipse, thresh, masked_pupil

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print("Iniciando prueba de preprocesamiento y segmentación de pupila...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Simulación de extracción de ROI de ojo
        h, w, _ = frame.shape
        cy, cx = h // 2, w // 2
        eye_roi_dummy = frame[cy-30:cy+30, cx-45:cx+45]

        # Pipeline A3
        scaled, preprocessed = preprocess_eye_roi(eye_roi_dummy, scale_factor=3.0)
        ellipse, thresh, result = segment_pupil(preprocessed)

        if result is not None:
            cv2.imshow("1. ROI Original Scaled", scaled)
            cv2.imshow("2. CLAHE + Blur", preprocessed)
            cv2.imshow("3. Threshold", thresh)
            cv2.imshow("4. Segmentación Pupilar", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
