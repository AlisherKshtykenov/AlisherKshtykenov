"""Тест-образец для батч-градиентного спуска (batch gradient descent). Проверяет эталон и attempts/batch_gd_attempt_NN.py."""
import numpy as np
import pytest

IMPL = ("src/optimizers/gradient_descent/batch_gd.py", "BatchGD")   # что тестируем: файл-эталон и имя класса в нём


def test_one_step_known_value(impl):
    """Известное значение: 1.0 - 0.1 * 2.0 = 0.8."""
    [x] = impl(learning_rate=0.1).step([1.0], [2.0])           # step возвращает список новых параметров
    assert x == pytest.approx(0.8)


def test_arrays_and_order(impl):
    """Вход/выход: несколько параметров-массивов, порядок и формы сохраняются."""
    w, b = np.array([1.0, -2.0]), 0.5
    new_w, new_b = impl(learning_rate=0.5).step([w, b], [np.array([2.0, 2.0]), 1.0])
    assert np.allclose(new_w, [0.0, -3.0])                     # каждый вес сдвинулся против своего градиента
    assert new_b == pytest.approx(0.0)


def test_zero_lr_does_not_move(impl):
    """learning rate (скорость обучения) = 0 — параметры не меняются."""
    [x] = impl(learning_rate=0.0).step([3.0], [10.0])
    assert x == pytest.approx(3.0)


def test_converges_to_parabola_minimum(impl):
    """Поведение: на f(x) = (x - 3)² спуск приходит в минимум x = 3."""
    opt, x = impl(learning_rate=0.1), 0.0
    for _ in range(200):
        [x] = opt.step([x], [2 * (x - 3)])                    # градиент (gradient) параболы: 2(x - 3)
    assert x == pytest.approx(3.0, abs=1e-6)                   # abs — допустимая абсолютная погрешность
