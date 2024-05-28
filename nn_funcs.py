import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """Функция активации ReLU.

    Args:
        x: Входной массив.

    Returns:
        Массив после применения ReLU.
    """
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    """Функция активации Softmax.

    Args:
        x: Входной массив.

    Returns:
        Массив после применения Softmax.
    """
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Функция вычисления точности.

    Args:
        y_pred: Предсказанные метки.
        y_true: Истинные метки.

    Returns:
        Значение точности.
    """
    return np.mean(np.argmax(y_pred, axis=1) == y_true)

def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Функция вычисления потерь кросс-энтропии.

    Args:
        y_pred: Предсказанные вероятности.
        y_true: Истинные метки.

    Returns:
        Значение потерь.
    """
    n_samples = y_pred.shape[0]
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    if len(y_true.shape) == 1:
        confidences = y_pred_clipped[range(n_samples), y_true]
    elif len(y_true.shape) == 2:
        confidences = np.sum(y_pred_clipped * y_true, axis=1)

    negative_log_likehoods = -np.log(confidences)
    return np.mean(negative_log_likehoods)
