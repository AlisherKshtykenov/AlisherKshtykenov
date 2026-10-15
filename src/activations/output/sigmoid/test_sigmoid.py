"""Тест-образец для сигмоиды. Проверяет эталон и каждую попытку attempts/attempt_NN.py."""
import numpy as np
import pytest

IMPL = "sigmoid"                                       # что тестируем: имя в эталоне; эталон и попытки — по папке теста

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


def test_derivative_at_zero(get):
    derivative = get("sigmoid_derivative")
    assert derivative(0) == pytest.approx(0.25)


def test_derivative_sigmoid_two(get):
    derivative = get("sigmoid_derivative")
    assert derivative(2) == pytest.approx(0.105, abs=1e-3)


def test_derivative_symmetric(get):
    derivative = get("sigmoid_derivative")
    assert pytest.approx(derivative(2), abs=1e-3) == pytest.approx(derivative(-2), abs=1e-3)


def test_derivative_shape_and_range(get):
    derivative = get("sigmoid_derivative")
    z = np.linspace(-50, 50, 101)
    out = np.asarray(derivative(z))
    assert out.shape == z.shape
    assert np.all((out >= 0) & (out <= 0.25))


def test_derivative_metches_numerical(get):
    sigmoid_fun = get("sigmoid")
    derivative = get("sigmoid_derivative")
    h = 10**-5
    assert pytest.approx(derivative(2), abs=1e-3) == pytest.approx((sigmoid_fun(2+h)-sigmoid_fun(2-h))/(2*h), abs=1e-3)