import pandas as pd
import joblib
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from src.preprocessing import clean_data
from src.business_metric import calcular_ganancia

df = pd.read_csv('data/raw/fraud_dataset.csv')
df = clean_data(df)

X = df.drop(columns=['Fraude'])
y = df['Fraude']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

model = xgb.XGBClassifier(scale_pos_weight=10, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba > 0.3).astype(int)

metrics = {
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred)
}

joblib.dump(model, 'models/xgboost_model.pkl')

with open('reports/xgb_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

ganancia = calcular_ganancia(y_test, y_pred, X_test['Monto'])
print(f"Ganancia estimada XGB: ${ganancia:,.2f}")

from src.visualizations import plot_roc_curve_xgb
plot_roc_curve_xgb('models/xgboost_model.pkl', X_test, y_test)
