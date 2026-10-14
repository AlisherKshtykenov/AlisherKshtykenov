"""Синхронизирует трекер ROADMAP-ML с проектом (в обе стороны), рисует картинку прогресса и пересобирает README-отчёт.

Запуск из корня репозитория:  python scripts/update_progress.py
"""
import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")                                          # Agg (без окна) — рисовать сразу в файл, годится для GitHub Actions
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, to_rgb

ROOT = Path(__file__).resolve().parents[1]                     # корень репо (скрипт лежит в scripts/)
ORDER = ["eda", "preprocessing", "model", "activation", "loss", "optimizer", "metrics"]
FULL = 4                                                       # сколько повторений = 100% «осело»
CMAP = LinearSegmentedColormap.from_list(                      # плавная шкала: серый → тёмно-зелёный
    "progress", ["#ebedf0", "#9be9a8", "#40c463", "#30a14e", "#216e39"])
LABEL_TOP, TITLE_Y = -0.35, -8.6                               # где подписи разделов и названия этапов (над кубиками)

BASE_COLS = ["topic", "chapter", "step", "learned", "repeated", "readme", "experiment", "test_create", "test_run"]
SERVICE_COLS = ["slug", "path", "description", "header"]       # служебные столбцы — в конце таблицы
READY = {"ready", "готов", "done"}                             # статусы «тема изучена»
STEPS = {"model": "model", "activations": "activation", "losses": "loss",
         "optimizers": "optimizer", "metrics": "metrics"}      # первая папка пути → этап
ANALYSIS = {"analysis", "data_prep"}                           # старое и новое имя папки анализа данных — одно и то же
SKIP_DIRS = {"attempts", "__pycache__"}                        # черновики и кэш — не эталоны

BROKEN_COLOR = "#cf222e"                                       # значок «!» (тест сломан) — красный
MISSING_COLOR = "#8250df"                                      # значок «?» (нет статуса в шапке) — фиолетовый


# ───────────────────────────── обход проекта ─────────────────────────────

@dataclass
class Etalon:
    slug: str                                                  # slug (короткое имя) — имя папки метода или файла
    chapter: str                                               # раздел, например analysis/preprocessing/missing
    step: str                                                  # этап: model, loss, eda, ...
    path: str                                                  # путь от корня репо, например src/.../sigmoid.py
    method_dir: Path | None                                    # папка метода (у плоских файлов её нет)
    status: str | None = None                                  # статус из шапки, None — строки status нет
    description: str = ""
    header_path: str | None = None                             # path, записанный в шапке
    has_code: bool = False                                     # есть ли код кроме шапки и docstring (строки документации)


def normalize_chapter(parts):
    """data_prep/... → analysis/..., чтобы совпадало с таблицей."""
    parts = list(parts)
    if parts and parts[0] in ANALYSIS:
        parts[0] = "analysis"
    return "/".join(parts)


def step_of(chapter):
    """Этап по первой части пути раздела; None — папка не относится к роадмапу."""
    parts = chapter.split("/")
    if parts[0] == "analysis":
        return parts[1] if len(parts) > 1 and parts[1] in ("eda", "preprocessing") else None
    return STEPS.get(parts[0])


def read_header(text):
    """Читает шапку: подряд идущие строки-комментарии `# ключ: значение` в начале файла."""
    meta = {}
    for line in text.splitlines()[:10]:
        line = line.strip()
        if not line.startswith("#"):
            break                                              # шапка кончилась на первой не-комментарной строке
        key, sep, value = line.lstrip("#").partition(":")
        if sep:
            meta[key.strip().lower()] = value.strip()
    return meta


DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)   # функции и классы


def is_stub(node):
    """Заглушка: импорт, pass, `...`, docstring или функция/класс, внутри которых только заглушки."""
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass)):
        return True
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):   # docstring (строка документации) или `...`
        return True
    if isinstance(node, DEFS):
        return all(is_stub(n) for n in node.body)              # def mse(): pass — пусто; класс из пустых методов — пусто
    return False


def has_code(text):
    """True — есть хотя бы одна функция или класс с настоящим телом. Проверка через ast (синтаксическое дерево),
    поэтому шапка и комментарии не видны вовсе, а импорты, pass, `...`, docstring и пустые def/class не считаются."""
    try:
        body = ast.parse(text).body
    except SyntaxError:
        return True                                            # битый код — всё равно код
    return any(isinstance(node, DEFS) and not is_stub(node) for node in body)


def py_etalons_in(folder):
    return [p for p in folder.glob("*.py") if not p.name.startswith("_")]


def is_method_dir(folder):
    """Папка метода: внутри нет подпапок (кроме attempts/кэша) и нет других .py, кроме `<имя папки>.py`."""
    subdirs = [d for d in folder.iterdir() if d.is_dir() and d.name not in SKIP_DIRS]
    return not subdirs and all(p.stem == folder.name for p in py_etalons_in(folder))


def scan_project(root):
    """Эталоны (.py) и папки методов без .py. Понимает и `<method>/<method>.py`, и плоские `<chapter>/<method>.py`."""
    src, etalons, empty_dirs = root / "src", [], []
    for folder in sorted([src, *src.rglob("*")]):
        if not folder.is_dir() or SKIP_DIRS & set(folder.relative_to(src).parts):
            continue
        method = folder != src and is_method_dir(folder)
        chapter_dir = folder.parent if method else folder
        chapter = normalize_chapter(chapter_dir.relative_to(src).parts)
        step = step_of(chapter) if chapter else None
        if step is None:
            continue                                           # src/utils и прочее вне роадмапа
        files = py_etalons_in(folder)
        if method and not files:                               # папка метода заведена, а файла ещё нет
            empty_dirs.append(Etalon(slug=folder.name, chapter=chapter, step=step, path="", method_dir=folder))
        for py in files:
            text = py.read_text(encoding="utf-8", errors="replace")
            meta = read_header(text)
            etalons.append(Etalon(slug=py.stem, chapter=chapter, step=step,
                                  path=py.relative_to(root).as_posix(), method_dir=folder if method else None,
                                  status=meta.get("status"), description=meta.get("description", ""),
                                  header_path=meta.get("path"), has_code=has_code(text)))
    return etalons, empty_dirs


def slug_from_topic(topic):
    """`Logistic Regression (логистическая регрессия)` → `logistic_regression`."""
    name = topic.split("(")[0].lower()
    return re.sub(r"[^a-zа-яё0-9]+", "_", name).strip("_")


def readme_has_text(folder):
    """1 — README.md есть и в нём есть что-то кроме заголовков."""
    readme = folder / "README.md" if folder else None
    if not readme or not readme.is_file():
        return 0
    lines = [l.strip() for l in readme.read_text(encoding="utf-8", errors="replace").splitlines()]
    return int(any(l and not l.startswith("#") for l in lines))


def in_chapter(item_chapter, row_chapter):
    """Файл лежит в разделе строки или глубже (missing/imputation → missing)."""
    return item_chapter == row_chapter or item_chapter.startswith(row_chapter + "/")


def chapter_dir(root, chapter):
    """Папка раздела на диске: analysis/... может физически лежать в data_prep/."""
    head, *rest = chapter.split("/")
    for name in (["analysis", "data_prep"] if head == "analysis" else [head]):
        if (root / "src" / name).is_dir():
            return root / "src" / name / Path(*rest)
    return root / "src" / Path(chapter)


def new_file_path(root, chapter, slug, etalons):
    """Куда класть новый файл: если в разделе методы лежат папками — `<slug>/<slug>.py`, иначе `<slug>.py`."""
    folder = chapter_dir(root, chapter)
    by_folders = any(e.method_dir is not None for e in etalons if in_chapter(e.chapter, chapter))
    return folder / slug / f"{slug}.py" if by_folders else folder / f"{slug}.py"


def create_stub(root, py):
    """Пустой эталон с шапкой. Существующие файлы не трогаем никогда."""
    rel = py.relative_to(root).as_posix()
    if not py.exists():
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_text(f"# path: {rel}\n# status: in_progress\n# description:\n", encoding="utf-8")
    return rel


# ───────────────────────────── синхронизация таблицы ─────────────────────────────

def key(e):
    """Ключ занятости кандидата: путь файла или папка метода без файла."""
    return e.path or str(e.method_dir)


def pick(items, slug, row, claimed, step_wide):
    """Строго по равенству slug: в разделе строки (или глубже); step_wide — единственный такой slug на этапе."""
    free = [e for e in items if e.slug == slug and key(e) not in claimed]
    free.sort(key=lambda e: e.path != row["path"])             # одинаковый slug — сначала файл, уже записанный в строке
    if not step_wide:
        same = [e for e in free if in_chapter(e.chapter, row["chapter"])]
        return same[0] if same else None
    on_step = [e for e in free if e.step == row["step"]]
    return on_step[0] if len(on_step) == 1 else None          # knn есть в двух разделах → не угадываем


def sync_table(root, df, log, created, test_results=None):
    """Таблица ⇄ архитектура: сопоставляет по slug, создаёт недостающие файлы, дописывает новые темы, считает поля."""
    for col in BASE_COLS + SERVICE_COLS:
        if col not in df.columns:
            df[col] = ""
    extra = [c for c in df.columns if c not in BASE_COLS + SERVICE_COLS]
    df = df[BASE_COLS + extra + SERVICE_COLS].astype(object)   # служебные — в конец, лишние столбцы не теряем

    etalons, empty_dirs = scan_project(root)
    candidates = etalons + empty_dirs                          # файлы-эталоны и папки методов, где файла ещё нет
    claimed, found, match = set(), {}, {}

    # 1. строка → файл / папка метода — только по slug: сначала в своём разделе для всех строк, потом по этапу
    for step_wide in (False, True):
        for i, row in df.iterrows():
            if i in found:
                continue
            e = pick(candidates, row["slug"] or slug_from_topic(row["topic"]), row, claimed, step_wide)
            if e is not None:
                found[i] = e
                claimed.add(key(e))

    for i, row in df.iterrows():
        slug = row["slug"] or slug_from_topic(row["topic"])
        e = found.get(i)
        if e is None or not e.path:                            # файла нет → создаём пустой эталон с шапкой
            if e is not None:                                  # папка метода уже есть — кладём файл в неё
                py = e.method_dir / f"{slug}.py"
            elif row["slug"]:                                  # slug задан явно — заводим файл по chapter + slug
                py = new_file_path(root, row["chapter"], slug, etalons)
            else:                                              # slug выведен из названия и ни с чем не совпал
                log.append(f"WARN  строка «{row['topic']}» ({row['chapter']}): нет файла или папки со slug «{slug}» — "
                           f"впишите правильный slug в таблицу (файл не создан, чтобы не плодить лишнее)")
                continue
            created.append(create_stub(root, py))
            log.append(f"NEW   {created[-1]}: создан пустой эталон для «{row['topic']}»")
            e = Etalon(slug=slug, chapter=row["chapter"], step=row["step"], path=created[-1],
                       method_dir=py.parent if py.parent.name == slug else None, status="in_progress")
        match[i] = e
        claimed.add(e.path)
        df.at[i, "slug"] = slug                                # запоминаем slug — дальше сопоставление точное

    # 2. файл → строка: эталоны, которых нет в таблице
    new_rows = []
    for e in etalons + empty_dirs:                              # и папки методов без .py (deep_learning/.../resnet)
        if key(e) in claimed:
            continue
        if not e.path:                                         # папка метода есть, а файла нет → заводим файл в ней
            e.path = create_stub(root, e.method_dir / f"{e.slug}.py")
            e.status = "in_progress"
            created.append(e.path)
            log.append(f"NEW   {e.path}: создан пустой эталон в папке метода")
        new_rows.append({**{c: "" for c in df.columns}, "topic": e.slug, "chapter": e.chapter, "step": e.step,
                         "slug": e.slug, "repeated": 0, "test_run": 0})   # раздел — из пути файла
        match[len(df) + len(new_rows) - 1] = e
        log.append(f"INFO  {e.path}: новая тема «{e.slug}» добавлена в таблицу")
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows, columns=df.columns)], ignore_index=True).astype(object)

    # 3. поля по правилам
    tests = {p.name for p in (root / "tests").rglob("test_*.py")} if (root / "tests").is_dir() else set()
    for i, row in df.iterrows():
        e = match.get(i)
        slug = row["slug"] or slug_from_topic(row["topic"])
        if e is None:                                          # файла нет и создать не смогли
            learned, header, path, desc, method_dir = 0, "", "", "", None
        else:
            if e.status is None:                               # шапки нет или нет строки status
                header = "missing"
                log.append(f"WARN  {e.path}: статус отсутствует (нет строки `# status:` в шапке)")
            else:
                header = "ok"
            if not e.has_code:
                learned = 0                                    # пустой файл = 0, даже со статусом
            else:
                learned = 1 if (e.status or "").lower() in READY else 0.5
            if e.header_path and e.header_path != e.path:
                log.append(f"WARN  {e.path}: path в шапке ({e.header_path}) не совпадает с реальным путём")
            path, desc, method_dir = e.path, e.description, e.method_dir
        exp_dir = root / "experiments" / slug
        df.at[i, "learned"] = learned
        df.at[i, "readme"] = readme_has_text(method_dir)
        df.at[i, "experiment"] = int(exp_dir.is_dir() and any(exp_dir.glob("*.ipynb")))
        df.at[i, "test_create"] = int(f"test_{slug}.py" in tests)
        if test_results is not None:                           # --pytest: test_run по факту прогона, теста нет → 0
            df.at[i, "test_run"] = test_results.get(f"test_{slug}.py", 0)
        df.at[i, "path"], df.at[i, "description"], df.at[i, "header"] = path, desc, header

    log.append(f"INFO  тем в таблице: {len(df)}, с файлом: {len(match)}, без файла: {len(df) - len(match)}, "
               f"создано файлов: {len(created)}")
    return df


# ───────────────────────────── pytest ─────────────────────────────

def run_pytest(root, log):
    """Гоняет каждый tests/**/test_<slug>.py отдельно. Возвращает {имя файла: 1 — упал, 0 — прошёл}."""
    results = {}
    for test in sorted((root / "tests").rglob("test_*.py")):
        rel = test.relative_to(root).as_posix()
        run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", rel],
                             cwd=root, capture_output=True, text=True, timeout=600)
        failed = int(run.returncode not in (0, 5))             # 5 — в файле не нашлось тестов, это не поломка
        results[test.name] = max(results.get(test.name, 0), failed)   # одноимённые тесты: упал хоть один → 1
        last = (run.stdout.strip().splitlines() or ["(нет вывода)"])[-1]
        log.append(f"{'FAIL ' if failed else 'INFO '} {rel}: pytest — {last}")
    return results


def disk_report(root, df):
    """По этапам: тем в таблице, эталонов на диске (файлы + папки методов), строк без файла, файлов без строки."""
    etalons, empty_dirs = scan_project(root)
    on_disk = etalons + empty_dirs
    paths = set(df["path"])
    rows = []
    for step in ORDER:
        g = df[df["step"] == step]
        items = [e for e in on_disk if e.step == step]
        rows.append({"этап": step, "тем в таблице": len(g), "эталонов на диске": len(items),
                     "строк без файла": int((g["path"] == "").sum()),
                     "файлов без строки": sum(1 for e in items if (e.path or "") not in paths)})
    return pd.DataFrame(rows)


# ───────────────────────────── сводка (для README и ноутбука) ─────────────────────────────

def numeric(df):
    """Копия таблицы с числами вместо текста + колонки todo (сколько задач не сделано) и share (доля повторений)."""
    df = df.copy()
    for col in ["learned", "repeated", "readme", "experiment", "test_create", "test_run"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["todo"] = ((df["learned"] < 1).astype(int) + (df["readme"] == 0) +    # сколько из четырёх задач не сделано
                  (df["experiment"] == 0) + (df["test_create"] == 0))
    df["share"] = df["repeated"].clip(upper=FULL) / FULL                      # повторения → доля от 0 до 1
    return df


def stage_summary(df):
    """Проценты по этапам: сколько тем, сколько изучено / в работе / в планах, готовность и закреплённость."""
    d = numeric(df)
    rows = []
    for step in ORDER + ["всего"]:
        g = d if step == "всего" else d[d["step"] == step]
        if g.empty:
            continue
        rows.append({"этап": step, "тем": len(g), "изучено": int((g["learned"] == 1).sum()),
                     "в работе": int((g["learned"] == 0.5).sum()), "в планах": int((g["learned"] == 0).sum()),
                     "готовность, %": round(g["learned"].mean() * 100),     # среднее learned: 1 / 0.5 / 0
                     "повторения, %": round(g["share"].mean() * 100)})     # как на картинке: repeated / 4
    return pd.DataFrame(rows)


# ───────────────────────────── картинка ─────────────────────────────

def render_tracker(df, out_path, scale=0.7):
    df = numeric(df)

    columns, chapters, steps = [], [], []
    for step in ORDER:
        start = len(columns)
        for chapter, grp in df[df["step"] == step].groupby("chapter", sort=False):
            chapters.append((len(columns), chapter.split("/")[-1]))
            columns.append(grp.to_dict("records"))
        steps.append((start, len(columns) - 1, step))          # первый и последний столбец этапа
        columns.append(None)                                   # отступ между этапами
    columns.pop()

    n_cols, n_rows = len(columns), max(len(c) for c in columns if c)
    legend_y = n_rows + 1.3                                    # строка легенды — под контейнерами
    s = scale
    x_max = max(n_cols + 3.2, draw_legend(None, x=0, y=legend_y, s=s))   # справа место под градусник (и легенду, если она длиннее)
    fig, ax = plt.subplots(figsize=((x_max + 0.4) * 0.33 * s + 0.3, (n_rows + 12) * 0.33 * s))

    for first, last, step in steps:                            # контейнеры этапов + названия
        ax.add_patch(FancyBboxPatch((first - 0.15, TITLE_Y - 1.0), last - first + 1.3, n_rows - TITLE_Y + 1.1,
                                    boxstyle="round,pad=0,rounding_size=0.5",
                                    facecolor="#f6f8fa", edgecolor="#d0d7de", linewidth=1, zorder=0))
        ax.text((first + last + 1) / 2, TITLE_Y, step.upper(), ha="center", va="center",
                fontsize=10 * s, fontweight="bold", color="#24292f")

    for x, name in chapters:                                   # подписи разделов — вплотную над своими кубиками
        ax.text(x + 0.5, LABEL_TOP, name, rotation=90, ha="center", va="bottom",
                fontsize=8.5 * s, color="#57606a")

    for x, col in enumerate(columns):                          # кубики со слоями
        for y, t in enumerate(col or []):
            draw_cube(ax, x, y, t["share"], todo=t["todo"],
                      missing=t["header"] == "missing", broken=t["test_run"] == 1, s=s)

    draw_legend(ax, x=0, y=legend_y, s=s)

    ax.set_xlim(-0.4, x_max)
    ax.set_ylim(legend_y + 1.0, TITLE_Y - 1.3)                 # ось перевёрнута: сверху этапы и подписи, ниже кубики и легенда
    ax.set_aspect("equal")
    ax.axis("off")

    progress = df["share"].mean()                              # общий прогресс по всему роадмапу
    ax.set_title(f"ROADMAP-ML · прогресс {progress:.0%}", loc="left",
                 fontsize=14 * s, fontweight="bold", color="#24292f", pad=4)

    draw_thermometer(ax, x=n_cols + 0.9, top=TITLE_Y - 1.0, bottom=n_rows + 0.1, s=s)   # градусник справа, на всю высоту контейнеров

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white",
                metadata={"Software": None})                   # без версии matplotlib — файл не меняется зря
    plt.close(fig)


def hatch_color(face):
    """Цвет штриховки под фон: на светлом — тёмная, на тёмном — светлая."""
    r, g, b = to_rgb(face)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b           # luminance (яркость) по формуле sRGB
    return (0.14, 0.16, 0.18, 0.55) if luminance > 0.5 else (1, 1, 1, 0.75)


def draw_badge(ax, cx, cy, sign, color, s):
    """Маленький значок-кружок с символом (как бейдж на иконке)."""
    ax.add_patch(Circle((cx, cy), 0.2, facecolor=color, edgecolor="white", linewidth=0.6 * s, zorder=4))
    ax.text(cx, cy + 0.01, sign, ha="center", va="center", fontsize=6 * s,
            fontweight="bold", color="white", zorder=5)


def draw_cube(ax, x, y, share, todo=0, missing=False, broken=False, s=0.7):
    """Кубик темы: цвет (повторения) → тонкая штриховка (не сделано) → значки «?» и «!» в разных углах."""
    x0, y0, size = x + 0.08, y + 0.08, 0.84
    face = CMAP(float(share))                                  # float: CMAP(1) с целым — это индекс, а не 100%
    cube = FancyBboxPatch((x0, y0), size, size, boxstyle="round,pad=0,rounding_size=0.18",
                          facecolor=face, edgecolor="none", zorder=1)
    ax.add_patch(cube)

    if todo:                                                   # 1. тонкая диагональная решётка, 4 уровня густоты
        step = size / (todo + 1.5)                             # чем больше несделанного, тем чаще линии
        color = hatch_color(face)
        for c in np.arange(-size, size, step) + step / 2:      # линии под 45° в обе стороны → решётка «x»
            for sign in (1, -1):
                xs = np.array([x0, x0 + size])
                ys = (y0 + c + (xs - x0)) if sign > 0 else (y0 + size - c - (xs - x0))
                line, = ax.plot(xs, ys, color=color, linewidth=0.35 * s, zorder=2, solid_capstyle="butt")
                line.set_clip_path(cube)                       # обрезаем по скруглённому кубику

    if missing:                                                # 2. «?» в левом верхнем углу — нет статуса в шапке
        draw_badge(ax, x0 + 0.2, y0 + 0.2, "?", MISSING_COLOR, s)
    if broken:                                                 # 3. «!» в правом верхнем углу — тест сломан
        draw_badge(ax, x0 + size - 0.2, y0 + 0.2, "!", BROKEN_COLOR, s)


def draw_legend(ax, x, y, s):
    """Компактная легенда в одну строку: штриховка, «!», «?». Возвращает правый край.
    С ax=None ничего не рисует — только считает ширину (нужно заранее для размера картинки)."""
    def cube(px, **layers):
        if ax is not None:
            draw_cube(ax, px, y, layers.pop("share", 0), s=s, **layers)

    def label(px, text):
        if ax is not None:
            ax.text(px, y + 0.5, text, ha="left", va="center", fontsize=8 * s, color="#57606a")
        return px + len(text) * 0.22 + 1.2                     # грубая оценка ширины текста в клетках + отступ

    for k in range(1, 5):                                      # четыре уровня штриховки на сером фоне
        cube(x + k - 1, todo=k)
    x = label(x + 4.2, "не сделано 1–4 из: learned, readme, experiment, test")
    cube(x, share=0.5, broken=True)
    x = label(x + 1.2, "тест сломан (test_run = 1)")
    cube(x, share=0.5, missing=True)
    return label(x + 1.2, "нет статуса в шапке (header = missing)")


def draw_thermometer(ax, x, top, bottom, s, width=0.75):
    """Шкала-градусник: вертикальная трубка со скруглёнными краями и градиентом внутри."""
    x0, x1, r = x - width / 2, x + width / 2, width / 2

    tube = FancyBboxPatch((x0, top), width, bottom - top,            # трубка ровно на высоту контейнеров
                          boxstyle=f"round,pad=0,rounding_size={r}",  # rounding_size = половина ширины → полукруглые концы
                          facecolor="none", edgecolor="#d0d7de", linewidth=1, zorder=3)
    ax.add_patch(tube)
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)                 # столбик значений от 0 до 1
    img = ax.imshow(gradient, cmap=CMAP, origin="lower", aspect="auto",
                    extent=[x0, x1, bottom, top], zorder=2)          # 0% внизу, 100% наверху
    img.set_clip_path(tube)                                          # обрезаем градиент по форме трубки

    for p in (0, 0.25, 0.5, 0.75, 1):                                # риски и подписи, как на градуснике
        y = (bottom - r) + p * ((top + r) - (bottom - r))            # от низа до верха, не заходя в скругления
        ax.plot([x1 + 0.12, x1 + 0.4], [y, y], color="#8c959f", linewidth=0.8, zorder=3)
        ax.text(x1 + 0.52, y, f"{p:.0%}", va="center", ha="left", fontsize=8 * s, color="#57606a")


# ───────────────────────────── README-отчёт ─────────────────────────────

START, END = "<!-- progress:start -->", "<!-- progress:end -->"


def cell(text):
    """Текст для ячейки markdown-таблицы: без переносов и с экранированным «|»."""
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def file_link(path):
    return f"[`{Path(path).name}`]({path})" if path else "—"


def topics_table(g):
    """Таблица тем: тема · этап · файл · где применяется."""
    if g.empty:
        return "_Пока пусто._\n"
    lines = ["| Тема | Этап | Файл | Где применяется |", "|---|---|---|---|"]
    for _, t in g.iterrows():
        lines.append(f"| {cell(t['topic'])} | {t['step']} | {file_link(t['path'])} | {cell(t['description']) or '—'} |")
    return "\n".join(lines) + "\n"


def build_readme(df):
    """Личный отчёт: картинка, проценты, изучено / в работе / планы, проблемы. Всё — из таблицы."""
    d = numeric(df)
    summary = stage_summary(df)
    total = summary.iloc[-1]
    out = ["# ROADMAP-ML", "",
           "Мой личный трекер пути в ML: что уже изучено, что в работе, что впереди и где сейчас проблемы.",
           "Этот файл собирается сам из `roadmap_tracker.csv` скриптом `scripts/update_progress.py` — руками не править.",
           "", START, "![ROADMAP-ML progress](assets/progress.png)", END, "",
           "## Прогресс", "",
           f"**Готовность: {total['готовность, %']}%** · изучено {total['изучено']} из {total['тем']} тем · "
           f"в работе {total['в работе']} · в планах {total['в планах']} · "
           f"закреплено повторениями: {total['повторения, %']}%", "",
           "| Этап | Тем | Изучено | В работе | В планах | Готовность | Повторения |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in summary.iterrows():
        name = f"**{r['этап']}**" if r["этап"] == "всего" else r["этап"]
        out.append(f"| {name} | {r['тем']} | {r['изучено']} | {r['в работе']} | {r['в планах']} | "
                   f"{r['готовность, %']}% | {r['повторения, %']}% |")
    out += ["", "_Готовность — среднее `learned` (изучено = 1, в работе = 0.5, в планах = 0). "
                "Повторения — `repeated` / 4, по ним красятся кубики на картинке._", ""]

    out += ["## Изучено", "", topics_table(d[d["learned"] == 1]),
            "## В работе", "", "_Код уже есть, но статус в шапке ещё не `ready`._", "",
            topics_table(d[d["learned"] == 0.5]),
            "## Планы", "", "_Что ещё предстоит выучить: файл заведён, но кода пока нет (или файла нет совсем)._", ""]
    plans = d[d["learned"] == 0]
    for step in ORDER:
        g = plans[plans["step"] == step]
        if g.empty:
            continue
        out.append(f"<details><summary><b>{step}</b> — {len(g)}</summary>\n")
        for chapter, cg in g.groupby("chapter", sort=False):
            out.append(f"- **{chapter.split('/')[-1]}**: " + ", ".join(cell(t) for t in cg["topic"]))
        out += ["", "</details>", ""]

    no_header = d[d["header"] == "missing"]
    broken = d[d["test_run"] == 1]
    no_file = d[d["path"] == ""]
    out += ["## Проблемы", ""]
    if no_header.empty and broken.empty and no_file.empty:
        out += ["_Проблем нет._", ""]
    if not broken.empty:
        out += [f"**`!` Сломан тест — {len(broken)}**", ""]
        out += [f"- {cell(t['topic'])} — {file_link(t['path'])}" for _, t in broken.iterrows()] + [""]
    if not no_header.empty:
        out += [f"<details><summary><b><code>?</code> Нет строки <code>status</code> в шапке — {len(no_header)}</b>"
                f"</summary>\n"]                               # список длинный — сворачиваем
        out += [f"- {cell(t['topic'])} — {file_link(t['path'])}" for _, t in no_header.iterrows()]
        out += ["", "</details>", ""]
    if not no_file.empty:
        out += [f"**Нет файла: slug не совпал ни с одним файлом или папкой — {len(no_file)}** "
                f"(впишите правильный `slug` в таблицу)", ""]
        out += [f"- {cell(t['topic'])} — `{t['chapter']}`" for _, t in no_file.iterrows()] + [""]
    return "\n".join(out).rstrip() + "\n"


# ───────────────────────────── запуск ─────────────────────────────

def main(root=ROOT, verbose=True, pytest=False):
    """Полный цикл. Возвращает таблицу, строки лога и список созданных файлов (удобно для ноутбука).
    pytest=True — прогнать тесты и проставить test_run по факту, иначе test_run берётся из таблицы."""
    root = Path(root).resolve()
    csv_path, log_path = root / "roadmap_tracker.csv", root / "logs" / "progress.log"
    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)   # всё как текст — ручные поля не портим
    log, created = [], []
    test_results = run_pytest(root, log) if pytest else None
    df = sync_table(root, df, log, created, test_results)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path.write_text(f"# update_progress {stamp}\n" + "\n".join(log) + "\n", encoding="utf-8")   # перезапись при каждом запуске

    render_tracker(df, root / "assets" / "progress.png")
    (root / "README.md").write_text(build_readme(df), encoding="utf-8")

    if verbose:
        for line in log:
            if not line.startswith("INFO"):
                print(line)
        print(disk_report(root, df).to_string(index=False))
        print(f"Готово: roadmap_tracker.csv, logs/progress.log, assets/progress.png, README.md "
              f"(создано файлов: {len(created)}, предупреждений: {sum(l.startswith('WARN') for l in log)})")
    return df, log, created


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="корень проекта (по умолчанию — репозиторий)")
    parser.add_argument("--pytest", action="store_true", help="прогнать pytest и проставить test_run автоматически")
    args = parser.parse_args()
    main(args.root, pytest=args.pytest)
