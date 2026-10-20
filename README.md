# Sistema de Monitoreo Visual en Cabina (TEL-328)

Proyecto de procesamiento digital de imágenes para la detección en tiempo real de signos de fatiga y alteraciones físicas en conductores.

## Estructura del repositorio
- `requirements.txt`: Dependencias del proyecto (OpenCV, MediaPipe, NumPy, Scikit-learn).
- `test_setup.py`: Script de validación de entorno de ejecución.
- `inspect_data.py`: Script de verificación de rutas y análisis de resolución óptica.
- `data/`: Estructuración local de datasets (particiones de UTA-RLDD, YawDD y secuencias de prueba de pupilometría).

## Instalación y ejecución
```bash
# Crear y activar entorno virtual
python -m venv venv
.\venv\Scripts\Activate  # En Windows

# Instalar librerías
pip install -r requirements.txt

# Validar entorno
python test_setup.py
