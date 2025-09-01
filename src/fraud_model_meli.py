import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, roc_curve,
    precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

CSV_PATH = "data/fraud_dataset.csv"
REPORTS_DIR = "reports"

os.makedirs(REPORTS_DIR, exist_ok=True)

def parse_monto(x):
    if pd.isna(x): return np.nan
    s = str(x).replace(",", "")
    try: return float(s)
    except: return np.nan

def profit_at_threshold(y_true, p_fraud, threshold, amounts, gain_rate=0.25, loss_rate=1.0):
    approve = p_fraud < threshold
    profit = np.where(
        approve & (y_true == 1), -loss_rate * amounts,
        np.where(approve & (y_true == 0), gain_rate * amounts, 0.0)
    )
    return profit.sum()

def main():
    # 1) Datos
    df = pd.read_csv(CSV_PATH)
    if "Monto" in df.columns:
        df["Monto"] = df["Monto"].apply(parse_monto)
    else:
        raise ValueError("No se encontró la columna 'Monto' en el CSV.")
    if "Fraude" not in df.columns:
        raise ValueError("No se encontró la columna 'Fraude' en el CSV.")

    y = df["Fraude"].astype(int)
    X = df.drop(columns=["Fraude"])

    # 2) Preprocesamiento
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    num_tf = Pipeline([("imp", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False))
    ])

    prep = ColumnTransformer([("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)])

    # 3) Split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    amounts_te = X_te["Monto"].to_numpy()

    # 4) Modelos
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42), 
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1, class_weight="balanced_subsample"
    ),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    }

    rows = []
    best = None

    for name, clf in models.items():
        pipe = Pipeline([("prep", prep), ("model", clf)])
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)[:, 1]

        # Métricas
        auc = roc_auc_score(y_te, p)
        ap = average_precision_score(y_te, p)

        # Barrido de umbrales por ganancia
        thresholds = np.linspace(0.0, 0.9, 181)
        profits = np.array([profit_at_threshold(y_te.to_numpy(), p, t, amounts_te) for t in thresholds])
        idx = int(np.argmax(profits))
        t_best = float(thresholds[idx])
        prof_best = float(profits[idx])

        rows.append({
            "model": name,
            "roc_auc": auc,
            "avg_precision": ap,
            "best_threshold": t_best,
            "best_profit": prof_best,
            "profit_at_t_0.2": float(profit_at_threshold(y_te.to_numpy(), p, 0.2, amounts_te))
        })

        # Reportes
        y_pred_best = (p >= t_best).astype(int)
        cm = confusion_matrix(y_te, y_pred_best, labels=[0, 1])
        cm_df = pd.DataFrame(cm,
            index=["Actual_NoFraude", "Actual_Fraude"],
            columns=["Pred_NoFraude", "Pred_Fraude"])
        cm_df.to_csv(f"{REPORTS_DIR}/confusion_matrix_{name}.csv", index=True)

        with open(f"{REPORTS_DIR}/classification_report_{name}.txt", "w") as f:
            f.write(classification_report(y_te, y_pred_best, digits=4))

        # Figuras
        fpr, tpr, _ = roc_curve(y_te, p)
        prec, rec, _ = precision_recall_curve(y_te, p)

        plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--')
        plt.title(f"ROC - {name} (AUC={auc:.3f})"); plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.savefig(f"{REPORTS_DIR}/roc_curve.png", bbox_inches="tight"); plt.close()

        plt.figure(); plt.plot(rec, prec)
        plt.title(f"Precision-Recall - {name} (AP={ap:.3f})"); plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.savefig(f"{REPORTS_DIR}/pr_curve.png", bbox_inches="tight"); plt.close()

        plt.figure()
        plt.plot(thresholds, profits)
        plt.axvline(0.2, linestyle="--")
        plt.axvline(t_best, linestyle="--")
        plt.title("Ganancia vs. Umbral"); plt.xlabel("Umbral"); plt.ylabel("Ganancia total (validación)")
        plt.savefig(f"{REPORTS_DIR}/profit_curve.png", bbox_inches="tight"); plt.close()

    results = pd.DataFrame(rows).sort_values(by="best_profit", ascending=False)
    results.to_csv(f"{REPORTS_DIR}/model_results.csv", index=False)

    # Mejor modelo
    best_row = results.iloc[0].to_dict()
    with open(f"{REPORTS_DIR}/best_model.txt", "w") as f:
        f.write(str(best_row))

    # JSON compacto
    payload = {
        "best_overall": {
            "name": best_row["model"],
            "best_threshold": float(best_row["best_threshold"]),
            "best_profit": float(best_row["best_profit"])
        },
        "summaries": {
            r["model"]: {
                "roc_auc": float(r["roc_auc"]),
                "avg_precision": float(r["avg_precision"]),
                "best_threshold": float(r["best_threshold"]),
                "best_profit": float(r["best_profit"]),
                "profit_at_t_0_2": float(r["profit_at_t_0.2"])
            } for _, r in results.iterrows()
        }
    }
    pd.Series(payload).to_json(f"{REPORTS_DIR}/model_profit_summary.json")

    print("OK. Resultados en:", REPORTS_DIR)
    print(results)

if __name__ == "__main__":
    main()
