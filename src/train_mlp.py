import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from src.preprocessing import clean_data
from src.business_metric import calcular_ganancia

df = pd.read_csv('data/raw/fraud_dataset.csv')
df = clean_data(df)

X = df.drop(columns=['Fraude'])
y = df['Fraude']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=300)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
f1 = f1_score(y_test, y_pred)

joblib.dump(model, 'models/mlp_model.pkl')

with open('reports/mlp_metrics.json', 'w') as f:
    json.dump({"f1_score": f1}, f, indent=4)

ganancia = calcular_ganancia(y_test, y_pred, X_test['Monto'])
print(f"Ganancia estimada MLP: ${ganancia:,.2f}")
