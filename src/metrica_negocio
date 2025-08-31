def calcular_ganancia(y_true, y_pred, monto):
    ganancia = 0
    for real, pred, m in zip(y_true, y_pred, monto):
        if pred == 0 and real == 0:
            ganancia += 0.25 * m
        elif pred == 0 and real == 1:
            ganancia -= m
    return ganancia
