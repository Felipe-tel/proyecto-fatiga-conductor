# Sistema de Monitoreo Visual en Cabina (TEL-328)

Proyecto de procesamiento digital de imágenes para la detección en tiempo real de signos de fatiga y alteraciones físicas en conductores.

## Estructura del repositorio
- `requirements.txt`: Dependencias del proyecto (OpenCV, MediaPipe, NumPy, Scikit-learn, Matplotlib, Pandas).
- `test_setup.py`: Script de validación de entorno de ejecución.
- `inspect_data.py`: Script de verificación de rutas y análisis de resolución óptica.
- `main.py`: Punto de entrada principal para ejecutar el sistema integrado (monitoreo ocular + pose 3D de cabeza).
- `face_landmarks.py`: Script principal para captura de video, detección facial, extracción de ROIs, cálculo de métricas (EAR, PIR), pose 3D (Pitch, Yaw, Roll) y alertas de microsueño/cabeceo/distracción.
- `pupil_segmentation.py`: Módulo de preprocesamiento ocular y segmentación pupilar.
- `head_pose.py`: Módulo de estimación de pose e inclinación 3D de la cabeza mediante `cv2.solvePnP` (Pitch, Yaw, Roll y proyección del vector celeste 3D en la nariz).
- `a4/`: Módulo A4, bostezos (MAR) y movimientos oculares (fijaciones, sacadas, nistagmo). Se ejecuta dentro de `main.py` (alimenta el bostezo de A6) y de `face_landmarks.py`.
- `a4_run_video.py`, `a4_protocolo.py`, `a4_eval_yawdd.py`, `a4_pruebas.py`: A4 por separado, protocolo de validación guiado, evaluación en YawDD y pruebas sintéticas. Detalle en [docs/A4.md](docs/A4.md).
- `models/`: Directorio que almacena el modelo preentrenado `face_landmarker.task` de MediaPipe.
- `data/`: Estructuración local de datasets (particiones de UTA-RLDD, YawDD y secuencias de prueba de pupilometría).
- `evidencias/`: Carpeta generada automáticamente para almacenar capturas de pantalla de las pruebas.

## Instalación y ejecución
```bash
# Crear y activar entorno virtual
python -m venv venv
.\venv\Scripts\Activate  # En Windows

# Instalar librerías
pip install -r requirements.txt

# Validar entorno
python test_setup.py

# Ejecutar con cámara web en vivo
python main.py

# Ejecutar con un video de prueba 
python main.py --video "ruta/al/video.mp4"

# --cam elige la cámara (0 por defecto; probar 1 si abre otra)
# --salida guarda los indicadores de A4 al cerrar
python main.py --cam 1 --salida resultados/
```

### Panel A4 (bostezos y mirada)
En `main.py` (panel lateral, líneas "A4") y en `face_landmarks.py` (bajo las métricas de A3 y de la pose) aparece el estado de **A4**. En `main.py` el bostezo que recibe el clasificador A6 viene de A4:
- Los **primeros 5 s** son de calibración: mantener la boca cerrada (`--calib-a4 0` la desactiva).
- Durante los **30 s siguientes** se arma la línea base de la mirada del conductor; después se activan las alertas relativas.
- Alertas: `BOSTEZO`, `FATIGA: n bostezos en 15 min` (3 o más, criterio clínico de bostezo excesivo), `MIRADA FIJA` (fijación de más de 6 s), `EXPLORACION VISUAL REDUCIDA` y `OSCILACION OCULAR` (las dos últimas comparan con la línea base del conductor).
- Color: verde = normal, naranjo = aviso, rojo = alerta, celeste = calibrando.

```bash
# Pruebas automáticas de A4 (no necesitan cámara)
python a4_pruebas.py

# Protocolo guiado de validación (estímulo en pantalla, métricas por tramo)
python a4_protocolo.py --persona Nombre --out protocolo/
```

Para esta entrega final, la bitacora 2 se puede encontrar en el siguiente enlace del repositorio
