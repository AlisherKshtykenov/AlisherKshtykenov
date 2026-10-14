"""Тест-образец для сигмоиды. Проверяет эталон и каждую попытку attempts/sigmoid_attempt_NN.py."""
import numpy as np
import pytest

IMPL = ("src/activations/output/sigmoid.py", "sigmoid")       # что тестируем: файл-эталон и имя функции в нём


def test_zero_is_half(impl):
    """Известное значение: ноль баллов = монетка."""
    assert impl(0) == pytest.approx(0.5)                       # approx (примерно) — сравнение дробных чисел с допуском


def test_known_value(impl):
    """Ещё одно точное значение: sigmoid(ln 3) = 1 / (1 + 1/3) = 0.75."""
    assert impl(np.log(3)) == pytest.approx(0.75)


def test_shape_and_range(impl):
    """Вход и выход: массив той же формы, все значения в [0, 1], функция не убывает."""
    z = np.linspace(-50, 50, 101)                              # 101 число от -50 до 50
    out = np.asarray(impl(z))
    assert out.shape == z.shape                                # shape (форма) не меняется
    assert np.all((out >= 0) & (out <= 1))                     # это вероятности
    assert np.all(np.diff(out) >= 0)                           # diff (разности соседей) ≥ 0 — монотонность


def test_symmetry(impl):
    """Свойство: sigmoid(-z) = 1 - sigmoid(z)."""
    z = np.array([-3.0, -0.5, 0.0, 2.0, 7.0])
    assert np.allclose(impl(-z), 1 - impl(z))                  # allclose (все близки) — поэлементное сравнение


def test_no_overflow(impl):
    """Огромные по модулю входы не ломают расчёт: без переполнения (overflow) и nan."""
    with np.errstate(over="raise"):                            # любое переполнение станет ошибкой и уронит тест
        out = np.asarray(impl([-1000.0, 1000.0]))
    assert np.allclose(out, [0.0, 1.0])
