import numpy as np

def create_neuron(input_size: int, output_size: int) -> np.ndarray:
    """Создает нейрон с заданным размером входа и выхода.

    Args:
        input_size: Размерность входных данных.
        output_size: Размерность выходных данных.

    Returns:
        Матрица весов нейрона.
    """
    weights = np.random.randn(input_size, output_size)
    return weights


def create_combination(hidden_size: int, neuron_population_size: int) -> np.ndarray:
    """Создает комбинацию индексов нейронов.

    Args:
        hidden_size: Количество нейронов в комбинации (размер скрытого слоя).
        neuron_population_size: Размер популяции нейронов.

    Returns:
        Массив индексов нейронов.
    """
    return np.random.randint(0, neuron_population_size, size=hidden_size)

def mutate_neuron(weights: np.ndarray, mutation_rate: float) -> np.ndarray:
    """Мутирует веса нейрона с заданной вероятностью.

    Args:
        weights: Матрица весов нейрона.
        mutation_rate: Вероятность мутации.

    Returns:
        Мутированная матрица весов.
    """
    mutation_mask = np.random.rand(*weights.shape) < mutation_rate
    weights += 0.1 * np.random.randn(*weights.shape) * mutation_mask
    return weights

def mutate_combination(combination: np.ndarray,
                       mutation_rate: float,
                       neuron_population_size: int) -> np.ndarray:
    """Мутирует комбинацию индексов нейронов.

    Args:
        combination: Массив индексов нейронов.
        mutation_rate: Вероятность мутации.
        neuron_population_size: Размер популяции нейронов.

    Returns:
        Мутированный массив индексов.
    """
    for i in range(len(combination)):
        if np.random.rand() < mutation_rate:
            combination[i] = np.random.randint(0, neuron_population_size)
    return combination
