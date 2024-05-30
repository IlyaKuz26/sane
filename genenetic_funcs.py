from typing import Tuple

import numpy as np


def create_chromosome(input_size: int, hidden_size: int, output_size: int) -> np.ndarray:
    """
    Создает хромосому, содержащую веса для всех нейронов.

    Args:
        input_size: Размерность входных данных.
        hidden_size: Количество нейронов в скрытом слое.
        output_size: Размерность выходных данных.

    Returns:
        Одномерный массив, представляющий хромосому.
    """
    hidden_weights_size = input_size * hidden_size
    output_weights_size = hidden_size * output_size
    chromosome = np.random.randn(hidden_weights_size + output_weights_size)
    return chromosome

def split_chromosome(chromosome: np.ndarray,
                     input_size: int,
                     hidden_size: int,
                     output_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Разделяет хромосому на веса для скрытого и выходного слоя.

    Args:
        chromosome: Хромосома, содержащая все веса.
        input_size: Размерность входных данных.
        hidden_size: Количество нейронов в скрытом слое.
        output_size: Размерность выходных данных.

    Returns:
        Кортеж из двух матриц весов: для скрытого и выходного слоя.
    """
    hidden_weights_size = input_size * hidden_size
    hidden_weights = chromosome[:hidden_weights_size].reshape(input_size, hidden_size)
    output_weights = chromosome[hidden_weights_size:].reshape(hidden_size, output_size)
    return hidden_weights, output_weights

def create_combination(population_size: int, hidden_size: int) -> np.ndarray:
    """
    Создает комбинацию индексов нейронов.

    Args:
        population_size: Размер популяции.
        hidden_size: Количество нейронов в комбинации (размер скрытого слоя).
        
    Returns:
        Массив индексов нейронов.
    """
    return np.random.randint(0, population_size, size=hidden_size)

def mutate_neuron(chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
    """
    Мутирует веса хромосомы с заданной вероятностью.

    Args:
        weights: Матрица весов нейрона.
        mutation_rate: Вероятность мутации.

    Returns:
        Мутированная матрица весов.
    """
    mutation_mask = np.random.rand(chromosome.size) < mutation_rate
    chromosome += 0.1 * np.random.randn(chromosome.size) * mutation_mask
    return chromosome

def mutate_combination(combination: np.ndarray,
                       mutation_rate: float,
                       population_size: int) -> np.ndarray:
    """
    Мутирует комбинацию индексов нейронов.

    Args:
        combination: Массив индексов нейронов.
        mutation_rate: Вероятность мутации.
        population_size: Размер популяции.

    Returns:
        Мутированный массив индексов.
    """
    for i in range(len(combination)):
        if np.random.rand() < mutation_rate:
            combination[i] = np.random.randint(0, population_size)
    return combination
