# path: src/losses/classification/log_loss/log_loss.py
# status: in_progress
# description:

# src/losses/classification/log_loss/log_loss.py

import numpy as np


def log_loss(y_true, y_pred, eps=1e-12):
    """Средняя ошибка вероятностей для бинарной классификации."""
    y_true = np.asarray(y_true, dtype=float)        # правильный ответ: 0 или 1б превращям в массив
    y_pred = np.asarray(y_pred, dtype=float)        # вероятности модели: от 0 до 1
    y_pred = np.clip(y_pred, eps, 1 - eps)          # eps (эпсилон, очень маленькое число) - не даем вероятности стать 0
    losses = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))     # ошибка для каждого объекта отдельно
    return losses.mean()    # mean (среднее) - одна общая цифра на весь набор


if __name__ == "__main__":
    print(log_loss([1], [0.99]))    # ~0.01  — уверенно и правильно, почти без штрафа
    print(log_loss([1], [0.5]))     # ~0.69  — монетка, средний штраф
    print(log_loss([1], [0.01]))    # ~4.6   — уверенно и неправильно, огромный штраф
    print(log_loss([1, 0], [1, 0])) # ~0     — без ошибки и без nan, clip сработал