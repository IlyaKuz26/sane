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
