```markdown
# Custom ML Metrics Implementation

Repozytorium zawiera własne implementacje metryk oceny modeli ML.

## 📊 Dostępne metryki

### Metryki Regresji (`regression.py`)
```python
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum((y_true - y_pred)**2)
    return 1 - (ss_res / ss_total)
```

**Status implementacji:**
- [x] MAE
- [x] RMSE
- [x] R²
- [ ] Obsługa przypadków brzegowych

### Walidacja (`validation.py`)
```python
def train_test_split(X, y, test_size=0.2):
    # TODO: Zaimplementuj losowy podział
    return X_train, X_test, y_train, y_test
```

**Status implementacji:**
- [x] Szkielet funkcji
- [ ] Losowe tasowanie
- [ ] Stratyfikacja

## 🛠 Jak użyć

1. Zaimportuj potrzebne metryki:
```python
from regression import mean_absolute_error
```

2. Oblicz metryki:
```python
mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.2f}")
```

## 🔍 Testowanie
```bash
# Uruchom testy jednostkowe
python -m pytest tests/test_metrics.py
```

## 📝 Plan rozwoju
- Dodanie metryk klasyfikacji
- Implementacja cross-validation
- Obsługa danych wielowymiarowych

## 📜 Licencja
Kod dostępny na licencji MIT. Więcej szczegółów w pliku LICENSE.
```