import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from genenetic_funcs import *
from nn_funcs import *


# Установка сида для генераторов случайных чисел
np.random.seed(42)

# Гипперпараметры
POPULATION_SIZE = 50  # Размер популяции
MUTATION_RATE = 0.01  # Вероятность мутации
CROSSOVER_RATE = 0.6  # Вероятность кроссинговера
HIDDEN_SIZE = 64  # Размер скрытого слоя
EPOCHS = 10  # Количество эпох обучения/популяций

# Загрузка датасета MNIST
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.astype(np.float32) / 255.0
y = y.astype(np.int64)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

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
    
    # Вывод точности лучшей особи в текущем поколении
    weights1, weights2 = create_nn(combinations[0], chromosomes[0], HIDDEN_SIZE)
    y_pred = predict(X_test, weights1, weights2)
    train_accuracy = accuracy(y_pred, y_test)
    print(f'Точность лучшей особи: {train_accuracy:.4f}')

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
      
# Тестирование лучшей ИНС на тестовых данных
weights1, weights2 = create_nn(combinations[0], chromosomes[0], HIDDEN_SIZE)
y_pred = predict(X_test, weights1, weights2)
test_accuracy = accuracy(y_pred, y_test)
print(f'Тестовая точность: {test_accuracy:.4f}')
