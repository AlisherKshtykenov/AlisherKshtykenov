"""Приводит шапки всех эталонов в src/ к виду `# path / # status / # description`.

Запуск из корня репозитория:  python scripts/add_headers.py
- нет шапки или нет строки status → ставится `status: todo`;
- в существующей шапке `path` обновляется на реальный путь, status и description сохраняются;
- код ниже шапки не меняется ни на символ. Повторный запуск ничего не меняет.
"""
import argparse
import re
from pathlib import Path

from update_progress import ROOT, scan_src                    # тот же обход src/, что и у трекера

PREAMBLE_RE = re.compile(r"^(#!|#.*coding[:=])")              # shebang (#!) и объявление кодировки — обязаны быть первыми
HEADER_RE = re.compile(r"^\s*#\s*(path|status|description)\s*:", re.IGNORECASE)   # строки нашей шапки


def fix_header(root, rel, status=None):
    """Пересобирает шапку файла. status — принудительный статус (иначе прежний или todo). Возвращает, что сделано."""
    py = root / rel
    lines = py.read_text(encoding="utf-8").splitlines(keepends=True)
    at = 0
    while at < len(lines) and at < 2 and PREAMBLE_RE.match(lines[at]):
        at += 1                                                # шапку ставим после shebang и coding, если они есть
    end = at
    while end < len(lines) and lines[end].lstrip().startswith("#"):
        end += 1                                               # верхний блок комментариев — там живёт шапка
    meta, rest_top = {}, []
    for line in lines[at:end]:
        m = HEADER_RE.match(line)
        if m:
            meta.setdefault(m.group(1).lower(), line.split(":", 1)[1].strip())
        else:
            rest_top.append(line)                              # чужие комментарии в шапке не теряем
    new_status = status or meta.get("status") or "todo"
    header = [f"# path: {rel}\n", f"# status: {new_status}\n", f"# description: {meta.get('description', '')}".rstrip() + "\n"]
    body = rest_top + lines[end:]
    if body and body[0].strip():
        header.append("\n")                                    # пустая строка между шапкой и тем, что было в файле
    new = "".join(lines[:at] + header + body)
    if new == "".join(lines):
        return "без изменений"
    py.write_text(new, encoding="utf-8")
    if "status" not in meta:
        return f"шапка добавлена (status: {new_status})"
    return "шапка обновлена" + (f" (status: {meta['status']} → {new_status})" if meta["status"] != new_status else "")


def main(root, overrides=None):
    """overrides — {slug или path: статус} для разового принудительного статуса."""
    overrides = overrides or {}
    done = {}
    for m in scan_src(root, []):
        status = overrides.get(m.path) or overrides.get(m.slug)
        done.setdefault(fix_header(root, m.path, status), []).append(m.path)
    for what, paths in sorted(done.items(), key=lambda kv: kv[0] == "без изменений"):
        print(f"{what}: {len(paths)}")
        if what != "без изменений":
            for path in paths:
                print("  ", path)
    return done


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="корень проекта (по умолчанию — репозиторий)")
    parser.add_argument("--status", nargs="*", default=[], metavar="SLUG=STATUS",
                        help="принудительно поставить статус, например sigmoid=in_progress")
    args = parser.parse_args()
    main(args.root.resolve(), dict(s.split("=", 1) for s in args.status))
