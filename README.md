# Sistema de Monitoreo Visual en Cabina (TEL-328)

Proyecto de procesamiento digital de imágenes para la detección en tiempo real de signos de fatiga y alteraciones físicas en conductores.

## Estructura del repositorio
- `requirements.txt`: Dependencias del proyecto (OpenCV, MediaPipe, NumPy, Scikit-learn, Matplotlib, Pandas).
- `test_setup.py`: Script de validación de entorno de ejecución.
- `inspect_data.py`: Script de verificación de rutas y análisis de resolución óptica.
- `face_landmarks.py`: Script principal para captura de video, detección facial (478 landmarks con MediaPipe), extracción de ROIs, cálculo de métricas (EAR, PIR) y alertas de microsueño.
- `pupil_segmentation.py`: Módulo de preprocesamiento ocular (interpolación bicúbica 3x, CLAHE, filtro gaussiano) y segmentación pupilar (umbralización adaptativa por percentiles, limpieza morfológica y ajuste de elipse).
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
python face_landmarks.py

# Ejecutar con un video de prueba 
python face_landmarks.py --video "ruta/al/video.mp4"
