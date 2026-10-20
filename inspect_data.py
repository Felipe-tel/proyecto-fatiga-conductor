import cv2
import os

print("--- REVISANDO CARPETAS DEL PROYECTO ---")

# Verificamos si creamos las carpetas
if os.path.exists("data/uta_rldd"):
    print("Carpeta uta_rldd encontrada.")
else:
    print("Falta crear carpeta uta_rldd")

if os.path.exists("data/yawdd"):
    print("Carpeta yawdd encontrada.")
else:
    print("Falta crear carpeta yawdd")

if os.path.exists("data/pupilas_test"):
    print("Carpeta pupilas_test encontrada.")
else:
    print("Falta crear carpeta pupilas_test")

print("\n--- PROBANDO CAMARA Y RESOLUCION ---")

# Abrir la camara web
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la camara web, revisando con tamaño por defecto...")
    ancho = 640
    alto = 480
else:
    ret, frame = cap.read()
    alto, ancho, canales = frame.shape
    cap.release()
    print("Camara web detectada correctamente.")

print("Resolucion de la camara:", ancho, "x", alto)

# Calculo aproximado del ojo a distancia normal de manejo
ancho_ojo = int(ancho * 0.06)
alto_ojo = int(alto * 0.04)

print("Tamano estimado del ojo en pixeles:", ancho_ojo, "x", alto_ojo)

if ancho_ojo < 45:
    print("Conclusion: El ojo se ve muy chico, hay que recortar y agrandar la imagen antes de medir la pupila.")
else:
    print("Conclusion: La resolucion alcanza para ver la pupila.")
