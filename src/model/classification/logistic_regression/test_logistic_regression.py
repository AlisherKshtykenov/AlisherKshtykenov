"""Тест-образец для логистической регрессии (logistic regression). Проверяет эталон и attempts/attempt_NN.py."""
import numpy as np
import pytest

IMPL = "LogisticRegression"                            # что тестируем: имя в эталоне; эталон и попытки — по папке теста


@pytest.fixture
def data():
    """Два облака точек, которые почти разделяются прямой. seed (зерно) фиксирован — данные одинаковы каждый раз."""
    rng = np.random.default_rng(0)                             # генератор случайных чисел с seed = 0
    X = np.vstack([rng.normal(-1.5, 1, (200, 2)), rng.normal(1.5, 1, (200, 2))])   # vstack — склеить строки
    y = np.array([0] * 200 + [1] * 200)
    idx = rng.permutation(len(y))                              # permutation (перестановка) — перемешать
    X, y = X[idx], y[idx]
    return X[:300], y[:300], X[300:], y[300:]                  # train (обучение) 300 и test (проверка) 100


def test_accuracy_above_threshold(impl, data):
    """Главное: на отложенных данных accuracy (доля верных ответов) выше 0.9."""
    X_train, y_train, X_test, y_test = data
    model = impl(learning_rate=0.1, n_iters=1000).fit(X_train, y_train)   # fit возвращает саму модель
    assert (model.predict(X_test) == y_test).mean() > 0.9


def test_outputs_shape_and_range(impl, data):
    """Вход/выход: predict_proba — вероятности в [0, 1] по одной на объект; predict — только 0 и 1."""
    X_train, y_train, X_test, _ = data
    model = impl(learning_rate=0.1, n_iters=200).fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test),)
    assert np.all((proba >= 0) & (proba <= 1))
    assert set(np.unique(model.predict(X_test))) <= {0, 1}     # <= для множеств (set) — «является подмножеством»


def test_loss_goes_down(impl, data):
    """Обучение идёт: лосс (loss, ошибка) в конце меньше, чем в начале."""
    X_train, y_train, _, _ = data
    model = impl(learning_rate=0.1, n_iters=300).fit(X_train, y_train)
    assert model.losses[-1] < model.losses[0]
