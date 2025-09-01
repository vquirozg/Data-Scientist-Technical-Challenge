
# Prevención de Fraude en Mercado Libre por Valeria Quiroz 

Proyecto técnico desarrollado como parte del proceso de selección para Data Scientist en MELI.

## Objetivo
Predecir transacciones fraudulentas maximizando la ganancia de negocio por medio de deteccion de fraude con algoritmos de Machine Learning:
Se parte de los siguientes supuestos

- Transacción legítima aprobada → +25% del monto.
- Transacción fraudulenta aprobada → −100% del monto.
- Transacción rechazada → 0.

##  Estructura
- `data/` → dataset con los datos otorgados por la entidad.
- `notebooks/` → exploración y experimentos (Jupyter/Colab).
- `src/` → script reproducible con pipeline ML.
- `reports/` →  métricas y resultados.

## Instalación y ejecucion del proyecto

Los pasos para ejecucion del scrip son> clonar el repositorio > luego instalar los requerimientos o librerias necesarias > ejecutar. Cada comando es mencionado a continuacion


Clonar el repositorio 
```bash
git clone https://github.com/vquirozg/Data-Scientist-Technical-Challenge.git
cd fraud_model_meli
pip install -r requirements.txt

#uso python src/fraud_model_meli.py
