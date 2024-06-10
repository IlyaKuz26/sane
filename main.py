import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from genenetic_funcs import *
from nn_funcs import *


def main(experiment=1, iterations=5):
    """
    Обучает SANE на датасете MNIST и визуализирует результаты.

    Args:
        experiment: Номер эксперимента
        iterations: Количество итераций генетического алгоритма.
    """
    # Гипперпараметры
    POPULATION_SIZE = 50  # Размер популяции
    MUTATION_RATE = 0.01  # Вероятность мутации
    CROSSOVER_RATE = 0.9  # Вероятность кроссинговера
    HIDDEN_SIZE = 64  # Размер скрытого слоя
    EPOCHS = 5  # Количество эпох обучения/популяций

    # Загрузка датасета MNIST
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int64)

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Инициализация массивов для хранения значений потерь и точности
    total_train_losses = np.zeros((iterations, EPOCHS))
    total_train_accuracies = np.zeros((iterations, EPOCHS))
    
    for k in range(iterations):
        print('\nИтерация', k + 1)

        # Создание начальной популяции хромосом
        chromosomes = np.array([create_chromosome(784, HIDDEN_SIZE, 10) for _ in
                                    range(POPULATION_SIZE)])

        # Создание начальной популяции комбинаций
        combinations = np.array([create_combination(HIDDEN_SIZE) for _ in range(POPULATION_SIZE)])

        # Обучение SANE
        for epoch in range(EPOCHS):
            print(f'Поколение {epoch + 1}/{EPOCHS}')

            # Оценка комбинаций и обновление приспособленностей хромосом
            chromosome_fitness = []

            for i in range(POPULATION_SIZE):
                # Формирование ИНС
                weights1, weights2 = create_nn(combinations[i], chromosomes[i], HIDDEN_SIZE)

                # Оценка ИНС
                y_pred = predict(X_train, weights1, weights2)
                loss = cross_entropy_loss(y_pred, y_train)
                chromosome_fitness.append(loss)

            # Сортировка комбинаций по приспособленности
            sorted_combinations = np.argsort(chromosome_fitness)
            chromosomes = [chromosomes[i] for i in sorted_combinations]
            combinations = [combinations[i] for i in sorted_combinations]
            
            # Вывод потерь и точности лучшей особи в текущем поколении
            weights1, weights2 = create_nn(combinations[0], chromosomes[0], HIDDEN_SIZE)
            y_pred = predict(X_test, weights1, weights2)
            train_accuracy = accuracy(y_pred, y_test)
            train_loss = chromosome_fitness[0]
            print(f'Потери лучшей особи: {train_loss:.4f}, Точность: {train_accuracy:.4f}')

            total_train_losses[k, epoch] = train_loss
            total_train_accuracies[k, epoch] = train_accuracy

            # Скрещивание хромосом и комбинаций
            new_chromosomes = []
            new_chromosomes = chromosomes.copy()
            new_combinations = []
            new_combinations = combinations.copy()

            for i in range(int(POPULATION_SIZE * 0.9)):
                parent1 = chromosomes[np.random.randint(0, int(POPULATION_SIZE * 0.25))]
                parent2 = chromosomes[np.random.randint(0, int(POPULATION_SIZE * 0.25))]
                child = crossover(parent1, parent2, CROSSOVER_RATE)
                new_chromosomes[-i - 1] = child

                parent1 = combinations[np.random.randint(0, int(POPULATION_SIZE * 0.25))]
                parent2 = combinations[np.random.randint(0, int(POPULATION_SIZE * 0.25))]
                child = crossover(parent1, parent2, CROSSOVER_RATE)
                new_combinations[-i - 1] = child

            chromosomes = new_chromosomes
            combinations = new_combinations

            # Мутация хромосом и комбинация
            for i in range(POPULATION_SIZE):
                chromosomes[i] = mutate_chromosome(chromosomes[i], MUTATION_RATE)
                combinations[i] = mutate_combination(combinations[i], MUTATION_RATE, HIDDEN_SIZE)
        
        best_combination = combinations[0]
        best_chromosome = chromosomes[0]

        # Тестирование лучшей ИНС на тестовых данных
        weights1, weights2 = create_nn(best_combination, best_chromosome, HIDDEN_SIZE)
        y_pred = predict(X_test, weights1, weights2)
        test_accuracy = accuracy(y_pred, y_test)
        print(f'Тестовая точность: {test_accuracy:.4f}')

        filepath = f'model/model_exp{experiment}_iter{k}.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump((best_combination, best_chromosome), f)
    
    # Усреднение по итерациям
    mean_train_losses = np.mean(total_train_losses, axis=0)
    mean_train_accuracies = np.mean(total_train_accuracies, axis=0)

    # Сохранение средних значений в файл
    filepath = f'plot/mean_values_exp{experiment}.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump((mean_train_losses, mean_train_accuracies), f)

    # Построение графиков
    plt.figure(figsize=(12, 5))

    x_axis = np.arange(1, EPOCHS + 1)

    # График усредненных потерь
    plt.subplot(1, 2, 1)
    plt.plot(x_axis, mean_train_losses) 
    plt.title('Изменение усредненных потерь')
    plt.xlabel('Поколение')
    plt.ylabel('Усредненные потери')
    plt.xticks(x_axis)

    # График усредненной точности
    plt.subplot(1, 2, 2)
    plt.plot(x_axis, mean_train_accuracies)
    plt.title('Изменение усредненной точности')
    plt.xlabel('Поколение')
    plt.ylabel('Усредненная точность')
    plt.xticks(x_axis)

    plt.show()

if __name__ == '__main__':
    main(experiment=1, iterations=5)
