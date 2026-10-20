Dataset: UTA Real-Life Drowsiness Dataset (UTA-RLDD)
Uso en el proyecto: Entrenamiento y validacion para deteccion de fatiga y somnolencia.

Estructura de particion:
Para evitar sesgo de sujeto (data leakage), la separacion se hace por participante y no por frames:
- Total participantes: 60 personas.
- Division: 5 grupos (folds) de 12 personas cada uno.
  * Fold 1: Sujetos 01 al 12
  * Fold 2: Sujetos 13 al 24
  * Fold 3: Sujetos 25 al 36
  * Fold 4: Sujetos 37 al 48
  * Fold 5: Sujetos 49 al 60

Estados etiquetados por participante:
- Alerta (0)
- Baja vigilancia (1)
- Somnolencia (2)

Formato de entrada: Video RGB (resolucion variable segun webcam/telefono).
Nota: Este dataset se usa para parpadeo y cabeceo, complementado con YawDD para bostezos.
