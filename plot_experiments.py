import pickle
from typing import List

from matplotlib import pyplot as plt
import numpy as np


def main(experiments: List[int]):
    """
    Загружает данные из нескольких экспериментов и строит графики изменений усредненных значений.

    Args:
        experiments: Список номеров экспериментов для загрузки.
    """
    plt.figure(figsize=(12, 5))

    # Загрузка и отображение данных для каждого эксперимента
    for exp_num in experiments:
        filepath = f'plot/mean_values_exp{exp_num}.pkl'
        with open(filepath, "rb") as f:
            mean_train_losses, mean_train_accuracies = pickle.load(f)

        x_axis = np.arange(1, len(mean_train_losses) + 1)
        
        plt.subplot(1, 2, 1)
        plt.plot(x_axis, mean_train_losses, label=f"Эксперимент {exp_num}")
        plt.title('Изменение усредненных потерь')
        plt.xlabel('Поколение')
        plt.ylabel('Усредненные потери')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(x_axis, mean_train_accuracies, label=f"Эксперимент {exp_num}")
        plt.title('Изменение усредненной точности')
        plt.xlabel('Поколение')
        plt.ylabel('Усредненная точность')
        plt.legend()

    plt.show()

if __name__ == '__main__':
    main([1, 2, 3, 4])
