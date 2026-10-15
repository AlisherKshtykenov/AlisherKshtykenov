# path: src/activations/output/sigmoid/sigmoid.py
# status: ready
# description: Превращает любое число в вероятность от 0 до 1


import numpy as np

def sigmoid(z):
    """Переводит любое число в вероятность от 0 до 1."""    # docString - строка документации
    z = np.asarray(z, dtype=float)      # asarray (как массив) - превращает вход в массив NumPy
    z = np.clip(z, -500, 500)    # clip (обрезать) - ограничиваем z, иначе exp(1000) переполнится
    return 1 / (1 + np.exp(-z))         # сама формула сигмоида


def sigmoid_derivative(z):
    """Производная сигмоида: насколько сильнее меняется ответ при небольшом сдвиге z."""
    sigmoid_z = sigmoid(z)      # считаем сигмоиду один раз, чтобы не вызывать exp дважды
    return sigmoid_z * (1 - sigmoid_z)      # формула σ(z)·(1 − σ(z)), поэлементно для каждого z