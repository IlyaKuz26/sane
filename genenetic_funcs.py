from typing import Tuple

import numpy as np


# Установка сида для генераторов случайных чисел
np.random.seed(42)

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

def create_combination(hidden_size: int) -> np.ndarray:
    """
    Создает комбинацию индексов нейронов.

    Args:
        hidden_size: Количество нейронов в комбинации (размер скрытого слоя).
        
    Returns:
        Массив индексов нейронов.
    """
    return np.random.randint(0, hidden_size, size=hidden_size)

def crossover(parent1: np.ndarray, parent2: np.ndarray,
              crossover_rate: float) -> np.ndarray:
    """
    Скрещивает две родительские особи.

    Args:
        parent1: Первая родительская особь.
        parent2: Вторая родительская особь.
        crossover_rate: Вероятность скрещивания.

    Returns:
        Потомок или первый родитель.
    """
    if np.random.rand() < crossover_rate:
        crossover_point = np.random.randint(1, parent1.size)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return child
    else:
        return parent1

def mutate_chromosome(chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
    """
    Мутирует веса хромосомы с заданной вероятностью.

    Args:
        chromosome: Хромосома с весами нейронов.
        mutation_rate: Вероятность мутации.

    Returns:
        Мутированная матрица весов.
    """
    mutation_mask = np.random.rand(chromosome.size) < mutation_rate
    chromosome += 0.1 * np.random.randn(chromosome.size) * mutation_mask
    return chromosome

def mutate_combination(combination: np.ndarray,
                       mutation_rate: float,
                       hidden_size: int) -> np.ndarray:
    """
    Мутирует комбинацию индексов нейронов.

    Args:
        combination: Массив индексов нейронов.
        mutation_rate: Вероятность мутации.
        hidden_size: Размер скрытого слоя.

    Returns:
        Мутированный массив индексов.
    """
    for i in range(len(combination)):
        if np.random.rand() < mutation_rate:
            combination[i] = np.random.randint(0, hidden_size)
    return combination
