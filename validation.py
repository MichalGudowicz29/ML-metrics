import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Dzieli dane na zbiór treningowy i testowy
    
    Parametry:
    X (np.array): Cechy
    y (np.array): Etykiety
    test_size (float): Proporcja zbioru testowego (0-1)
    random_state (int): Ziarno losowości
    
    Zwraca:
    (X_train, X_test, y_train, y_test)
    """
    if random_state:
        np.random.seed(random_state)
        
    # TODO: Zaimplementuj logikę podziału
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    split_idx = int(X.shape[0] * (1 - test_size))
    
    return (
        X[indices[:split_idx]], 
        X[indices[split_idx:]], 
        y[indices[:split_idx]], 
        y[indices[split_idx:]]
    )