import numpy as np

def mean_absolute_error(y_true, y_pred):
    """Oblicza średni błąd bezwzględny (MAE)"""
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    """Oblicza pierwiastek błędu średniokwadratowego (RMSE)"""
    return np.sqrt(np.mean((y_true - y_pred)**2))

def r2_score(y_true, y_pred):
    """Oblicza współczynnik determinacji R²"""
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum((y_true - y_pred)**2)
    return 1 - (ss_res / ss_total)

def mean_absolute_percentage_error(y_true, y_pred):
    """Oblicza średni procentowy błąd bezwzględny (MAPE)"""
    epsilon = np.finfo(np.float64).eps  # Zabezpieczenie przed dzieleniem przez 0
    return 100 * np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)))