import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

CSV_PATH = "MercadoLibre Inc. Data Scientist Hiring Test - Fraud Dataset  - Data.csv"

# --- Utilidades ---

def parse_monto(x):
    """Convierte campo Monto a float (elimina separadores de miles)."""
    if pd.isna(x):
        return np.nan
    s = str(x).replace(",", "")
    try:
        return float(s)
    except:
        return np.nan

def profit_at_threshold(y_true, p_fraud, threshold, amounts, gain_rate=0.25, loss_rate=1.0):
    """
    Calcula la ganancia total a un umbral dado.
    - Aprobada legítima: +25% monto
    - Aprobada fraudulenta: -100% monto
    - Rechazada: 0
    """
    approve = p_fraud < threshold
    profit = np.where(
        approve & (y_true == 1), -loss_rate * amounts,
        np.where(approve & (y_true == 0), gain_rate * amounts, 0.0)
    )
    return profit.sum()

# --- Pipeline principal ---

def main():
    # 1. Cargar dataset
    df = pd.read_csv(CSV_PATH)
    df["Monto"] = df["Monto"].apply(parse_monto)

    y = df["Fraude"].astype(int)
    X = df.drop(columns=["Fraude"])

    # 2. Definir tipos de variables
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    # 3. Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 4. Definir modelos
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42)
    }

    results = []
    best = None

    # 5. Entrenar y evaluar
    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", clf)])
        pipe.fit(X_train, y_train)
        p = pipe.predict_proba(X_test)[:, 1]

        # Métricas clásicas
        auc = roc_auc_score(y_test, p)
        ap = average_precision_score(y_test, p)

        # Barrido de umbrales para maximizar ganancia
        thresholds = np.linspace(0.0, 0.9, 181)
        profits = np.array([profit_at_threshold(y_test.to_numpy(), p, t, X_test["Monto"].to_numpy()) 
                            for t in thresholds])
        idx = int(np.argmax(profits))
        t_best = float(thresholds[idx])
        prof_best = float(profits[idx])

        results.append({
            "model": name,
            "roc_auc": auc,
            "avg_precision": ap,
            "best_threshold": t_best,
            "best_profit": prof_best,
            "profit_at_t_0.2": float(profit_at_threshold(y_test.to_numpy(), p, 0.2, X_test["Monto"].to_numpy()))
        })

        if best is None or prof_best > best["best_profit"]:
            best = {"name": name, "best_threshold": t_best, "best_profit": prof_best}

        # Guardar reportes
        y_pred_best = (p >= t_best).astype(int)
        cm = confusion_matrix(y_test, y_pred_best, labels=[0, 1])
        cm_df = pd.DataFrame(cm,
                             index=["Actual_NoFraude", "Actual_Fraude"],
                             columns=["Pred_NoFraude", "Pred_Fraude"])
        cm_df.to_csv(f"confusion_matrix_{name}.csv", index=True)

        with open(f"classification_report_{name}.txt", "w") as f:
            f.write(classification_report(y_test, y_pred_best, digits=4))

    # 6. Guardar resultados resumen
    results_df = pd.DataFrame(results).sort_values(by="best_profit", ascending=False)
    results_df.to_csv("model_results.csv", index=False)

    with open("best_model.txt", "w") as f:
        f.write(str(best))

    print("Resultados comparativos:\n", results_df)
    print("\nMejor modelo encontrado:\n", best)

if __name__ == "__main__":
    main()
