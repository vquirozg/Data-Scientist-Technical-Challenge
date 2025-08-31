import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.preprocessing import clean_data
from src.business_metric import calcular_ganancia

df = pd.read_csv('data/raw/fraud_dataset.csv')
df = clean_data(df)

X = df.drop(columns=['Fraude'])
y = df['Fraude']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

joblib.dump(model, 'models/random_forest.pkl')

with open('reports/rf_metrics.json', 'w') as f:
    json.dump(report, f, indent=4)

ganancia = calcular_ganancia(y_test, y_pred, X_test['Monto'])
print(f"Ganancia estimada RF: ${ganancia:,.2f}")

from src.visualizations import plot_feature_importance_rf
plot_feature_importance_rf('models/random_forest.pkl', X.columns)
