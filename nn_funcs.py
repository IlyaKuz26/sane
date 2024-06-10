from typing import Tuple

import numpy as np
from pyvis.network import Network

from genenetic_funcs import split_chromosome


def relu(x: np.ndarray) -> np.ndarray:
    """
    Функция активации ReLU.

    Args:
        x: Входной массив.

    Returns:
        Массив после применения ReLU.
    """
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Функция активации Softmax.

    Args:
        x: Входной массив.

    Returns:
        Массив после применения Softmax.
    """
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Функция для вычисления точности.

    Args:
        y_pred: Предсказанные метки.
        y_true: Истинные метки.

    Returns:
        Значение точности.
    """
    return np.mean(np.argmax(y_pred, axis=1) == y_true)

def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Функция вычисления потерь кросс-энтропии.

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

def create_nn(combination: np.ndarray,
              chromosome: np.ndarray,
              hidden_size: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Формирует нейронную сеть из комбинации индексов и хромосомы.

    Args:
        combination: Массив индексов нейронов.
        chromosome: Массив весов всех нейронов.
        hidden_size: Размер скрытого слоя.
 
    Returns:
        Кортеж из двух матриц весов: для скрытого и выходного слоя.
    """
    hidden_weights, output_weights = split_chromosome(chromosome, 784,
                                                      hidden_size, 10)

    weights1 = np.hstack([hidden_weights[:, i].reshape(-1, 1) for i in combination])
    weights2 = np.vstack([output_weights[i, :] for i in combination])
    return weights1, weights2

def predict(X: np.ndarray, weights1: np.ndarray,
            weights2: np.ndarray) -> np.ndarray:
    """
    Выполняет предсказание с помощью нейронной сети.

    Args:
        X: Входные данные.
        weights1: Матрица весов для скрытого слоя.
        weights2: Матрица весов для выходного слоя.

    Returns:
        Матрица предсказанных вероятностей.
    """
    hidden_layer = relu(np.dot(X, weights1))
    output_layer = softmax(np.dot(hidden_layer, weights2))
    return output_layer

def visualize_network(chromosome: np.ndarray, combination: np.ndarray,
                     hidden_size: int, filepath: str):
    """
    Визуализирует структуру нейронной сети с помощью pyvis.

    Args:
        chromosome: Хромосома, содержащая веса сети.
        combination: Комбинация индексов нейронов.
        hidden_size: Размер скрытого слоя.
        filepath: Путь к файлу для сохранения визуализации.
    """
    net = Network(height='900px', width='1200px', directed=True)
    net.options.layout = {'hierarchical': {'enabled': True, 'direction': 'LR', 'sortMethod': 'directed'}}
    net.toggle_physics(False)

    # Добавление входных нейронов
    input_neurons = 784 // 8  # Уменьшаем число нейронов для улучшения читаемости
    for i in range(input_neurons): 
        net.add_node(f'input_{i}', label=f'Вх{i}', color='#00FF00')

    # Добавление скрытых нейронов
    for i, hidden_idx in enumerate(combination):
        net.add_node(f'hidden_{hidden_idx}', label=f'Скр{hidden_idx}', color='#FFFF00')

    # Добавление выходных нейронов
    for i in range(10):
        net.add_node(f'output_{i}', label=f'Вых{i}', color='#FF0000')

    # Добавление связей
    hidden_weights, output_weights = split_chromosome(chromosome, 784, hidden_size, 10)
    for i, hidden_idx in enumerate(combination):
        for j in range(input_neurons):
            weight = hidden_weights[j, hidden_idx]
            net.add_edge(f'input_{j}', f'hidden_{hidden_idx}', 
                         value=abs(weight),
                         color='#FF0000' if weight > 0 else '#0000FF',
                         title=f'{weight:.2f}')
        for j in range(10):
            weight = output_weights[hidden_idx, j]
            net.add_edge(f'hidden_{hidden_idx}', f'output_{j}', 
                         value=abs(weight),
                         color='#FF0000' if weight > 0 else '#0000FF',
                         title=f'{weight:.2f}')
            
    # Сохранение визуализации в HTML-файл
    net.save_graph(filepath)
