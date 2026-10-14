"""Общие настройки тестов: каждый тест прогоняется на эталоне из src/ и на всех попытках из attempts/.

Как подключить в тест-файле:
    IMPL = ("src/activations/output/sigmoid.py", "sigmoid")   # файл-эталон и имя функции/класса в нём
    def test_zero(impl):                                      # impl — эта функция из эталона или из попытки
        assert impl(0) == 0.5

Попытка = attempts/<slug>_attempt_NN.py в папке метода, подраздела или этапа (ищем в таком порядке).
То же правило поиска — в scripts/update_progress.py (attempt_files), менять вместе.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]                     # корень репозитория


def attempt_files(etalon):
    """Все попытки к эталону: <slug>_attempt_*.py из attempts/ метода → подраздела → этапа."""
    slug = etalon.stem
    method_dir = etalon.parent if etalon.parent.name == slug else None   # папка метода есть только у .../<slug>/<slug>.py
    subsection = etalon.parent.parent if method_dir else etalon.parent
    found = {}
    for folder in (method_dir, subsection, subsection.parent):
        if folder is not None:
            for path in sorted((folder / "attempts").glob(f"{slug}_attempt_*.py")):
                found.setdefault(path.name, path)              # одноимённая попытка — берём самую «близкую»
    return list(found.values())


def load(path, name):
    """Загружает файл как модуль (module) и достаёт из него функцию или класс по имени."""
    spec = importlib.util.spec_from_file_location(f"impl_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)                            # выполняем файл — как import, но по пути
    return getattr(module, name)


def pytest_generate_tests(metafunc):
    """Размножает каждый тест с параметром impl: [etalon] + по одному запуску на каждую попытку."""
    if "impl" not in metafunc.fixturenames:
        return
    rel, _ = metafunc.module.IMPL
    etalon = ROOT / rel
    paths = [etalon] + attempt_files(etalon)
    metafunc.parametrize("impl", [pytest.param(p, id="etalon" if p == etalon else p.stem) for p in paths],
                         indirect=True)                        # indirect — значение готовит фикстура (fixture) ниже


@pytest.fixture
def impl(request):
    """Функция/класс из эталона или попытки. Сломанная попытка падает здесь — только её запуск, не весь файл."""
    _, name = request.module.IMPL
    return load(request.param, name)
