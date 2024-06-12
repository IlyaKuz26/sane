import numpy as np
from matplotlib import pyplot as plt
from pyvis.network import Network

from genenetic_funcs import split_chromosome


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


def plot_results(total_train_losses: np.ndarray, 
                  total_train_accuracies: np.ndarray, 
                  filepath: str):
    """
    Строит и сохраняет графики потерь и точности для всех итераций и усредненных значений.

    Args:
        total_train_losses: Массив потерь для каждой итерации и эпохи.
        total_train_accuracies: Массив точностей для каждой итерации и эпохи.
        filepath: Путь для сохранения графика.
    """
    plt.figure(figsize=(12, 5))

    x_axis = np.arange(1, total_train_losses.shape[1] + 1)

    # График потерь
    plt.subplot(1, 2, 1)
    for i in range(total_train_losses.shape[0]):
        plt.plot(x_axis, total_train_losses[i], label=f"Итерация {i + 1}")
    plt.plot(x_axis, np.mean(total_train_losses, axis=0),
             label="Среднее", linestyle='--', color='black')
    plt.title('Изменение потерь во время обучения')
    plt.xlabel('Поколение')
    plt.ylabel('Потери')
    plt.legend()

    # График точности
    plt.subplot(1, 2, 2)
    for i in range(total_train_accuracies.shape[0]):
        plt.plot(x_axis, total_train_accuracies[i], label=f"Итерация {i + 1}")
    plt.plot(x_axis, np.mean(total_train_accuracies, axis=0),
             label="Среднее", linestyle='--', color='black')
    plt.title('Изменение точности во время обучения')
    plt.xlabel('Поколение')
    plt.ylabel('Точность')
    plt.legend()

    # Сохраняем и закрываем график
    plt.savefig(filepath)
    plt.close()
