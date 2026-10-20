import cv2
import numpy as np
import mediapipe as mp

print("--- REVISIÓN DE ENTORNO ---")
print("[OK] OpenCV instalado correctamente. Versión:", cv2.__version__)
print("[OK] NumPy instalado correctamente. Versión:", np.__version__)
print("[OK] MediaPipe instalado correctamente. Versión:", mp.__version__)

# Verificación de inicialización de módulos
from mediapipe.tasks import python as mp_tasks
print("[OK] Módulo mediapipe.tasks inicializado correctamente.")
print("[OK] Entorno base listo para procesamiento digital de imágenes.")