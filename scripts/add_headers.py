"""Разовый скрипт: вставляет шапку `# path / # status / # description` во все файлы-эталоны в src/.

Запуск из корня репозитория:  python scripts/add_headers.py
Повторный запуск безопасен: файлы, где уже есть строка `# status:`, не трогаются.
"""
import argparse
import re
from pathlib import Path

from update_progress import ROOT, scan_project                # тот же поиск эталонов, что и у трекера

STATUS_RE = re.compile(r"^\s*#\s*status\s*:", re.MULTILINE | re.IGNORECASE)   # строка `# status:` где угодно в файле
PREAMBLE_RE = re.compile(r"^(#!|#.*coding[:=])")              # shebang (#!) и объявление кодировки — обязаны быть первыми


def add_header(root, rel):
    """Вставляет шапку в начало файла (после shebang/кодировки, до docstring). True — шапка добавлена."""
    py = root / rel
    text = py.read_text(encoding="utf-8")
    if STATUS_RE.search(text):
        return False                                           # статус уже есть — не трогаем
    lines = text.splitlines(keepends=True)
    at = 0
    while at < len(lines) and at < 2 and PREAMBLE_RE.match(lines[at]):
        at += 1                                                # шапку ставим после shebang и coding, если они есть
    header = f"# path: {rel}\n# status: in_progress\n# description:\n"
    if lines[at:] and lines[at].strip():
        header += "\n"                                         # пустая строка между шапкой и тем, что было в файле
    py.write_text("".join(lines[:at]) + header + "".join(lines[at:]), encoding="utf-8")   # код ниже не меняется ни на символ
    return True


def main(root):
    etalons, _ = scan_project(root)
    added, skipped = [], []
    for e in etalons:
        (added if add_header(root, e.path) else skipped).append(e.path)
    print(f"Шапка добавлена: {len(added)}")
    for path in added:
        print("  +", path)
    print(f"Пропущено (строка status уже есть): {len(skipped)}")
    for path in skipped:
        print("  =", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="корень проекта (по умолчанию — репозиторий)")
    main(parser.parse_args().root.resolve())
