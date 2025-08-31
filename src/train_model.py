import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.preprocessing import preprocess_pipeline

# Cargar datos
df = pd.read_csv('data/fraud_dataset.csv')  # O data/processed/ si ya limpiaste

# Preprocesar
X, y = preprocess_pipeline(df, target_column='is_fraud')

# Dividir
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Entrenar
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluar
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

# Guardar modelo
joblib.dump(model, 'models/random_forest_balanced.pkl')

# Guardar métricas
with open('reports/rf_metrics.json', 'w') as f:
    json.dump(report, f, indent=4)

print(" Modelo y métricas guardados correctamente.")
