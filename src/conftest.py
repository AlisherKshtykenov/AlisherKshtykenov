"""Общие настройки тестов тем: каждый test_<тема>.py прогоняется на эталоне и на каждой попытке.

Тема — папка src/<этап>/<подраздел>/<тема>/:
    <тема>.py              — эталон
    attempts/attempt_NN.py — попытки (повторения)
    test_<тема>.py         — тест; в нём строка IMPL = "имя" — функция или класс, которые проверяем

    IMPL = "sigmoid"
    def test_zero(impl):          # impl — sigmoid из эталона или из попытки
        assert impl(0) == 0.5

Тот же поиск попыток — в scripts/update_progress.py (attempt_files), менять вместе.
"""
import importlib.util
from pathlib import Path

import pytest


def topic_files(test_path):
    """Эталон и попытки темы — по папке, где лежит тест."""
    topic = Path(test_path).resolve().parent
    return topic / f"{topic.name}.py", sorted((topic / "attempts").glob("attempt_*.py"))


def load(path, name):
    """Загружает файл как модуль (module) и достаёт из него функцию или класс по имени."""
    spec = importlib.util.spec_from_file_location(f"impl_{path.parent.parent.name}_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)                            # выполняем файл — как import, но по пути
    return getattr(module, name)


def pytest_generate_tests(metafunc):
    """Размножает каждый тест с параметром impl: [etalon] + по одному запуску на каждую попытку."""
    if "impl" not in metafunc.fixturenames:
        return
    etalon, attempts = topic_files(metafunc.module.__file__)
    params = [pytest.param(etalon, id="etalon")] + [pytest.param(p, id=p.stem) for p in attempts]
    metafunc.parametrize("impl", params, indirect=True)        # indirect — значение готовит фикстура (fixture) ниже


@pytest.fixture
def impl(request):
    """Функция/класс из эталона или попытки. Сломанная попытка падает здесь — только её запуск, не весь файл."""
    return load(request.param, request.module.IMPL)
