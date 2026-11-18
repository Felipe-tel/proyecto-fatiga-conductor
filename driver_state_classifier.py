from collections import deque
import numpy as np


class ClasificadorEstadoConductor:
    def __init__(self, ventana_frames=45):  # Subido de 30 a 45 frames (~2.5 - 3 segundos)
        self.historial_fatiga = deque(maxlen=ventana_frames)
        self.historial_fisiologico = deque(maxlen=ventana_frames)
        self.pir_basal = 0.38
        self.muestras_basal = []
        self.calibrado = False
        self.estado_actual = "ALERTA (NORMAL)"

    def actualizar_calibracion(self, pir):
        if pir is not None and 0.12 <= pir <= 0.75 and len(self.muestras_basal) < 30:
            self.muestras_basal.append(pir)
            if len(self.muestras_basal) == 30:
                self.pir_basal = float(np.median(self.muestras_basal))
                self.calibrado = True

    def clasificar(self, ear, perclos, duracion_cierre, mar, bostezo_activo, pitch, yaw, pir, sacadico):
        if not self.calibrado:
            self.actualizar_calibracion(pir)

        # 1. Evaluación de riesgo por fatiga / somnolencia (0 a 100)
        riesgo_f = 0.0
        # Solo suma puntaje ocular si el ojo lleva más de 0.8 segundos cerrado (evita parpadeos o risas)
        if duracion_cierre >= 0.8 or ear < 0.18:
            riesgo_f += 45.0
        if perclos > 28.0:
            riesgo_f += 25.0
        # Solo suma bostezo cuando supera los 30 frames con MAR > 0.68
        if bostezo_activo:
            riesgo_f += 25.0
        if pitch is not None and pitch > 18.0:
            riesgo_f += 25.0
        if yaw is not None and abs(yaw) > 32.0:
            riesgo_f += 20.0

        self.historial_fatiga.append(min(riesgo_f, 100.0))
        score_fatiga = float(np.mean(self.historial_fatiga))

        # 2. Evaluación de alteración fisiológica ocular (PIR + movimientos sacádicos)
        riesgo_s = 0.0
        if pir is not None:
            desviacion_pir = abs(pir - self.pir_basal)
            if desviacion_pir > 0.14:
                riesgo_s += 55.0
        if sacadico > 6.5:
            riesgo_s += 45.0

        self.historial_fisiologico.append(min(riesgo_s, 100.0))
        score_fisiologico = float(np.mean(self.historial_fisiologico))

        # 3. Máquina de estados con histéresis temporal ampliada
        alerta_critica = False
        color_ui = (0, 255, 0)

        if duracion_cierre >= 3.0 or score_fatiga >= 62.0:
            self.estado_actual = "SOMNOLENCIA CRITICA"
            color_ui = (0, 0, 255)
            alerta_critica = True
        elif score_fisiologico >= 55.0:
            self.estado_actual = "ALERTA FISIOLOGICA (PUPILA/MIRADA)"
            color_ui = (0, 0, 255)
            alerta_critica = True
        elif score_fatiga >= 32.0 or score_fisiologico >= 35.0:
            self.estado_actual = "BAJA VIGILANCIA (ADVERTENCIA)"
            color_ui = (0, 165, 255)
        else:
            self.estado_actual = "ALERTA (NORMAL)"
            color_ui = (0, 255, 0)

        return self.estado_actual, score_fatiga, score_fisiologico, alerta_critica, color_ui