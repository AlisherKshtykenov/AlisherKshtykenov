# path: src/model/classification/logistic_regression/logistic_regression.py
# status: in_progress
# description:

import numpy as np      # NumPy - библиотека для работы с массивами и матрицами

from activations.output.sigmoid import sigmoid
from losses.classification.log_loss import log_loss
from optimizers.gradient_descent.batch_gd import BatchGD

class LogisticRegression:       # объявляю класс (имя класса с большой буквы)

    def __init__(self, learning_rate=0.01 ,n_iters=1000):    # вызываем один раз при создании модели
        self.learning_rate = learning_rate  # learning rate (скорость обучения) - длина шага градиентного спуска
        self.n_iters = n_iters    # n_iters (число итераций) - сколько раз повторить цикл обучения
        self.weights = None     # weights (веса) - пока пустые, размер узнаем только в fit, увидев данные
        self.bias = None      # bias (смещение) - свободный член тоже появится в fit
        self.losses = []        # сюда будем записывать лосс на каждой итерации, чтобы потом нарисовать график


    def fit(self, X, y):
        X = np.asarray(X, dtype=float)      # признаки: строки - объекты, столбцы - признаки
        y = np.asarray(y, dtype=float)      # ответы 0 или 1
        n_samples, n_features = X.shape     # shape (форма) - сколько строк сколько столбцов

        self.weights = np.zeros(n_features)     # теперь зная число признаков -> создаем нулевые веса
        self.bias = 0.0     # смещаем точку к нулю
        optimize = BatchGD(self.learning_rate)      # создаем оптимизатор с нашим шагом

        for _ in range(self.n_iters):       # цикл обучения
            z = X @ self.weights + self.bias        # 1. баллы охотника:
            p = sigmoid(z)      # 2. баллы -> вероятности

            self.losses.append(log_loss(y, p))      # 3. записали ошибку для графиков

            error = p - y       # 4. насколько промахнулась по каждому объекту
            dw = X.T @ error / n_samples        # градиент по весам
            db = error.mean()       # градиент по смещению

            self.weights, self.bias = optimize.step(        # 5. шаг против градиента
                [self.weights, self.bias], [dw, db]
            )

        return self     # возвращаем саму модель - можно писать model.fit(X, y).predict(X)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        z = X @ self.weights + self.bias        # те же балы, что в fit, но веса уже обучены
        return sigmoid(z)       # вероятность класса 1 для каждого объекта

    def predict(self, X, threshold=0.5):        # threshold (порог) - "да / нет"
        return (self.predict_proba(X) >= threshold).astype(int)     # больше прога -> 1, иначе 0