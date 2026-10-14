"""Тест-образец для log loss (логарифмической функции потерь). Проверяет эталон и attempts/attempt_NN.py."""
import numpy as np
import pytest

IMPL = "log_loss"                                      # что тестируем: имя в эталоне; эталон и попытки — по папке теста


def test_coin_is_ln2(impl):
    """Известное значение: предсказание 0.5 стоит ln 2 ≈ 0.693."""
    assert impl([1], [0.5]) == pytest.approx(np.log(2))


def test_known_mean(impl):
    """Среднее по объектам: -(ln 0.8 + ln 0.7) / 2 для ответов [1, 0] и вероятностей [0.8, 0.3]."""
    expected = -(np.log(0.8) + np.log(0.7)) / 2               # считаем руками по формуле
    assert impl([1, 0], [0.8, 0.3]) == pytest.approx(expected)


def test_returns_scalar(impl):
    """Выход: одно число на весь набор, а не массив."""
    assert np.ndim(impl([1, 0, 1], [0.9, 0.2, 0.6])) == 0     # ndim (число измерений) = 0 — это скаляр (scalar)


def test_perfect_is_finite_zero(impl):
    """Идеальные вероятности 1 и 0: ошибка ~0 и без nan/inf (нужен clip)."""
    loss = impl([1, 0], [1.0, 0.0])
    assert np.isfinite(loss) and loss < 1e-6                   # isfinite (конечное) — не inf и не nan


def test_confident_mistake_costs_more(impl):
    """Уверенная ошибка штрафуется сильнее неуверенной."""
    assert impl([1], [0.01]) > impl([1], [0.4])
