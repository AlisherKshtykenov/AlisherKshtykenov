"""Собирает трекер ROADMAP-ML из архитектуры src/: таблица, карта прогресса, радар, история и README-визитка.

Запуск из корня репозитория:  python scripts/update_progress.py [--skip-tests]
Структура: src/<этап>/<подраздел>/<метод>.py  или  src/<этап>/<подраздел>/<метод>/<метод>.py
Повтор = файл attempts/<slug>_attempt_NN.py, прошедший тест tests/**/test_<slug>.py.
"""
import argparse
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")                                          # Agg (без окна) — рисовать сразу в файл, годится для GitHub Actions
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, to_rgb

ROOT = Path(__file__).resolve().parents[1]                     # корень репо (скрипт лежит в scripts/)
ORDER = ["eda", "preprocessing", "model", "activations", "losses", "optimizers", "metrics"]   # этапы = папки src/
FULL = 10                                                      # 10 засчитанных повторений = 100% цвета
CMAP = LinearSegmentedColormap.from_list(                      # плавная шкала: серый → тёмно-зелёный
    "progress", ["#ebedf0", "#9be9a8", "#40c463", "#30a14e", "#216e39"])
LABEL_TOP, TITLE_Y = -0.35, -8.6                               # где подписи разделов и названия этапов (над кубиками)

COLUMNS = ["stage", "subsection", "slug", "path", "description", "status", "learned",
           "repeated", "last_repeat", "score", "readme", "experiment", "test_create", "test_run"]
FROM_TESTS = ["repeated", "last_repeat", "score", "test_run"]   # считаются прогоном тестов; с --skip-tests — из старой таблицы
LEARNED = {"todo": 0, "in_progress": 0.5, "ready": 1, "готов": 1, "done": 1}   # статус → learned
TASKS = ["learned", "readme", "experiment", "test"]            # четыре задачи по теме
REQUIRED = {"model": ["learned", "readme", "experiment", "test"],               # обязательные задачи этапа
            "eda": ["learned", "readme", "experiment"],                         # эксперимент = применено в ноутбуке
            "preprocessing": ["learned", "readme", "experiment"],
            "activations": ["learned", "readme", "test"], "losses": ["learned", "readme", "test"],
            "optimizers": ["learned", "readme", "test"], "metrics": ["learned", "readme", "test"]}
SKIP_DIRS = {"attempts", "__pycache__"}                        # черновики и кэш — не методы
HISTORY_COLS = ["datetime", "всего"] + ORDER                   # progress_history.csv
IMPL_RE = re.compile(r"""^IMPL\s*=\s*\(\s*["']([^"']+)["']""", re.MULTILINE)   # строка IMPL = ("src/...py", ...) в тесте

BROKEN_COLOR = "#cf222e"                                       # значок «!» (тест сломан) — красный
MISSING_COLOR = "#8250df"                                      # значок «?» (нет статуса) — фиолетовый
IN_PROGRESS_COLOR = "#fb8500"                                  # рамка in_progress — оранжевая


# ───────────────────────────── обход src/ ─────────────────────────────

@dataclass
class Method:
    stage: str                                                 # этап — папка верхнего уровня: model, losses, ...
    subsection: str                                            # подраздел — папка над методом: classification, ...
    slug: str                                                  # slug (короткое имя) метода = имя файла / папки
    path: str                                                  # путь от корня репо: src/model/classification/knn/knn.py
    method_dir: Path | None                                    # папка метода (у плоских файлов её нет)
    status: str                                                # статус из шапки, "" — строки status нет
    description: str


def read_header(text):
    """Шапка: подряд идущие строки-комментарии `# ключ: значение` в начале файла."""
    meta = {}
    for line in text.splitlines()[:12]:
        line = line.strip()
        if not line.startswith("#"):
            break                                              # шапка кончилась на первой не-комментарной строке
        key, sep, value = line.lstrip("#").partition(":")
        if sep:
            meta.setdefault(key.strip().lower(), value.strip())
    return meta


def scan_src(root, log):
    """Все методы из src/<этап>/<подраздел>/. Неожиданная вложенность и папки без <метод>.py — в лог."""
    methods = []
    for stage in ORDER:
        stage_dir = root / "src" / stage
        if not stage_dir.is_dir():
            continue
        for sub in sorted(d for d in stage_dir.iterdir() if d.is_dir() and d.name not in SKIP_DIRS):
            for item in sorted(sub.iterdir()):
                if item.name in SKIP_DIRS or item.name.startswith(("_", ".")):
                    continue
                if item.is_file() and item.suffix == ".py":    # плоский метод: .../output/sigmoid.py
                    py, method_dir = item, None
                elif item.is_dir() and (item / f"{item.name}.py").is_file():   # папка метода: .../knn/knn.py
                    py, method_dir = item / f"{item.name}.py", item
                elif item.is_dir():
                    log.append(f"WARN  {item.relative_to(root).as_posix()}/: папка без {item.name}.py — "
                               f"не метод (лишняя вложенность?)")
                    continue
                else:
                    continue                                   # README.md и прочие файлы подраздела
                meta = read_header(py.read_text(encoding="utf-8", errors="replace"))
                methods.append(Method(stage=stage, subsection=sub.name, slug=py.stem,
                                      path=py.relative_to(root).as_posix(), method_dir=method_dir,
                                      status=meta.get("status", "").lower(), description=meta.get("description", "")))
    return methods


def readme_has_text(folder):
    """1 — README.md есть и в нём есть что-то кроме заголовков."""
    readme = folder / "README.md" if folder else None
    if not readme or not readme.is_file():
        return 0
    lines = [l.strip() for l in readme.read_text(encoding="utf-8", errors="replace").splitlines()]
    return int(any(l and not l.startswith("#") for l in lines))


# ───────────────────────────── тесты и повторы ─────────────────────────────

def attempt_files(etalon):
    """Попытки к эталону: <slug>_attempt_*.py из attempts/ метода → подраздела → этапа (как в tests/conftest.py)."""
    slug = etalon.stem
    method_dir = etalon.parent if etalon.parent.name == slug else None   # папка метода есть только у .../<slug>/<slug>.py
    subsection = etalon.parent.parent if method_dir else etalon.parent
    found = {}
    for folder in (method_dir, subsection, subsection.parent):
        if folder is not None:
            for path in sorted((folder / "attempts").glob(f"{slug}_attempt_*.py")):
                found.setdefault(path.name, path)              # одноимённая попытка — берём самую «близкую»
    return list(found.values())


def test_key(root, test):
    """Чей это тест: путь эталона из строки IMPL, иначе «slug:<этап>:<slug>» по имени и папке теста."""
    m = IMPL_RE.search(test.read_text(encoding="utf-8", errors="replace"))
    if m:
        return m.group(1)
    parts = test.relative_to(root / "tests").parts
    return f"slug:{parts[0] if len(parts) > 1 else ''}:{test.stem.removeprefix('test_')}"


def find_test(results, m):
    """Результат теста для метода: по пути эталона, иначе по этапу и slug."""
    return results.get(m.path) or results.get(f"slug:{m.stage}:{m.slug}") or results.get(f"slug::{m.slug}")


def junit_cases(xml):
    """Из JUnit-отчёта (XML) pytest: [(имя теста, id параметра или None, прошёл ли, пропущен ли)]."""
    cases = []
    for tc in ET.parse(xml).iter("testcase"):
        name = tc.get("name", "")
        bad = tc.find("failure") is not None or tc.find("error") is not None
        skipped = tc.find("skipped") is not None
        pid = re.search(r"\[(.+)\]$", name)                    # test_zero[sigmoid_attempt_01] → sigmoid_attempt_01
        cases.append((name, pid.group(1) if pid else None, not bad, skipped))
    return cases


def run_tests(root, log, run=True):
    """Гоняет каждый tests/**/test_*.py отдельно. {ключ эталона: {file, etalon_failed, passed_attempts, failed}}.
    run=False — только найти тесты, не запуская (для --skip-tests)."""
    results = {}
    tests = sorted((root / "tests").rglob("test_*.py")) if (root / "tests").is_dir() else []
    for test in tests:
        rel = test.relative_to(root).as_posix()
        res = {"file": rel, "etalon_failed": False, "passed_attempts": [], "failed": []}
        results[test_key(root, test)] = res
        if not run:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp) / "report.xml"
            proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                   f"--junitxml={xml}", rel], cwd=root, capture_output=True, text=True, timeout=600)
            cases = junit_cases(xml) if xml.exists() else []
        if not cases and proc.returncode not in (0, 5):        # 5 — тестов нет; другое без отчёта — тест не запустился
            cases = [(rel, None, False, False)]
        by_id = {}
        for name, pid, ok, skipped in cases:
            by_id.setdefault(pid, []).append((ok, skipped))
            if not ok:
                res["failed"].append((name, rel))
        res["etalon_failed"] = any(not ok for pid in (None, "etalon") for ok, _ in by_id.get(pid, []))
        res["passed_attempts"] = [pid for pid, rs in by_id.items()   # попытка засчитана: все её тесты прошли
                                  if pid not in (None, "etalon") and all(ok and not sk for ok, sk in rs)]
        n_attempts = sum(pid not in (None, "etalon") for pid in by_id)
        log.append(f"{'FAIL ' if res['failed'] else 'INFO '} {rel}: эталон {'упал' if res['etalon_failed'] else 'ок'}, "
                   f"попыток прошло {len(res['passed_attempts'])} из {n_attempts}")
    return results


def git_date(root, path):
    """Дата последнего коммита файла из git (не из файловой системы). Не закоммичен — сегодня."""
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%cs", "--", str(path)], cwd=root,
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out = ""
    return date.fromisoformat(out) if out else date.today()


def full_months(since, today):
    """Сколько полных месяцев прошло с даты since до today."""
    months = (today.year - since.year) * 12 + today.month - since.month
    return max(0, months - (today.day < since.day))            # месяц не «добит» — не считаем


def repeats(root, m, res, today):
    """Повторы метода: (сколько попыток прошло тест, дата последней, итог со штрафом за простой)."""
    if not res or not res["passed_attempts"]:
        return 0, "", 0                                        # нет попыток или ни одна не прошла — серый кубик
    by_stem = {p.stem: p for p in attempt_files(root / m.path)}
    paths = [by_stem[s] for s in res["passed_attempts"] if s in by_stem]
    if not paths:
        return 0, "", 0
    last = max(git_date(root, p) for p in paths)
    score = max(0, len(paths) - full_months(last, today))      # минус 1 за каждый полный месяц без нового повтора
    return len(paths), last.isoformat(), min(score, FULL)


# ───────────────────────────── таблица ─────────────────────────────

def build_table(root, old, log, results, ran_tests=True, today=None):
    """Пересобирает таблицу из src/. Без прогона тестов повторы и test_run берёт из старой таблицы по path."""
    today = today or date.today()
    kept = old.set_index("path")[[c for c in FROM_TESTS if c in old.columns]].to_dict("index") \
        if "path" in old.columns and len(old) else {}
    order = {p: i for i, p in enumerate(old["path"])} if "path" in old.columns else {}   # прежний порядок строк

    rows = []
    for m in scan_src(root, log):
        if m.status not in LEARNED:
            log.append(f"WARN  {m.path}: " + (f"неизвестный статус «{m.status}» (ожидается todo / in_progress / ready)"
                                              if m.status else "статус отсутствует (нет строки `# status:` в шапке)"))
        res = find_test(results, m)
        if ran_tests:
            repeated, last_repeat, score = repeats(root, m, res, today)
            test_run = int(bool(res and res["etalon_failed"]))
        else:
            k = kept.get(m.path, {})
            repeated, last_repeat, score, test_run = (k.get(c, d) for c, d in zip(FROM_TESTS, (0, "", 0, 0)))
        exp_dir = root / "experiments" / m.slug
        rows.append({"stage": m.stage, "subsection": m.subsection, "slug": m.slug, "path": m.path,
                     "description": m.description, "status": m.status,
                     "learned": LEARNED.get(m.status, 0),       # learned — только из статуса
                     "repeated": repeated, "last_repeat": last_repeat, "score": score,
                     "readme": readme_has_text(m.method_dir),
                     "experiment": int(exp_dir.is_dir() and any(exp_dir.glob("*.ipynb"))),
                     "test_create": int(res is not None),
                     "test_run": test_run})
    df = pd.DataFrame(rows, columns=COLUMNS).astype(object)
    df["learned"] = [f"{v:g}" for v in df["learned"]]          # 0 / 0.5 / 1, а не 0.0

    for path in sorted(set(order) - set(df["path"])):          # файл удалён → строка пропала (сообщаем)
        log.append(f"INFO  {path}: файла больше нет — строка убрана из таблицы")
    for path in df.loc[~df["path"].isin(order.keys()), "path"]:
        log.append(f"INFO  {path}: новая тема в таблице")

    # порядок: этап → подраздел (как раньше в таблице) → метод (как раньше), новые — в конец по алфавиту
    first_sub = {}
    for i, (stage, sub, path) in enumerate(zip(df["stage"], df["subsection"], df["path"])):
        k = order.get(path, len(order) + i)
        first_sub[(stage, sub)] = min(first_sub.get((stage, sub), k), k)   # подраздел — по самой ранней своей строке
    df["_k"] = [(ORDER.index(s), first_sub[(s, sub)], order.get(p, len(order)), p)
                for s, sub, p in zip(df["stage"], df["subsection"], df["path"])]
    return df.sort_values("_k").drop(columns="_k").reset_index(drop=True)


# ───────────────────────────── сводка (для README, истории и ноутбука) ─────────────────────────────

def numeric(df):
    """Копия таблицы с числами + undone (сколько ОБЯЗАТЕЛЬНЫХ для этапа задач не сделано) и share (доля цвета)."""
    df = df.copy()
    for col in ["learned", "repeated", "score", "readme", "experiment", "test_create", "test_run"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    done = {"learned": df["learned"] >= 1, "readme": df["readme"] == 1,
            "experiment": df["experiment"] == 1, "test": df["test_create"] == 1}   # сделана ли каждая задача
    df["undone"] = [sum(not done[task][i] for task in REQUIRED.get(stage, TASKS))   # только обязательные для этапа
                    for i, stage in zip(df.index, df["stage"])]
    df["share"] = df["score"].clip(upper=FULL) / FULL                         # итог повторов со штрафом → доля 0..1
    return df


def stage_summary(df):
    """Проценты по этапам: сколько тем, сколько todo / in_progress / ready, готовность и закреплённость."""
    d = numeric(df)
    rows = []
    for stage in ORDER + ["всего"]:
        g = d if stage == "всего" else d[d["stage"] == stage]
        if g.empty:
            continue
        rows.append({"этап": stage, "тем": len(g), "ready": int((g["learned"] == 1).sum()),
                     "in_progress": int((g["learned"] == 0.5).sum()), "todo": int((g["learned"] == 0).sum()),
                     "готовность, %": round(g["learned"].mean() * 100, 1),  # среднее learned: 1 / 0.5 / 0
                     "повторения, %": round(g["share"].mean() * 100, 1)})  # как на картинке: repeated / 4
    return pd.DataFrame(rows)


def append_history(root, df, log):
    """Дописывает строку в progress_history.csv, только если проценты изменились с прошлого раза."""
    path = root / "progress_history.csv"
    summary = stage_summary(df).set_index("этап")["готовность, %"]
    row = {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           **{c: summary.get(c, 0.0) for c in HISTORY_COLS[1:]}}
    if path.exists():
        hist = pd.read_csv(path, encoding="utf-8-sig")
        if len(hist) and all(abs(float(hist.iloc[-1][c]) - float(row[c])) < 1e-9 for c in HISTORY_COLS[1:]):
            log.append("INFO  progress_history.csv: проценты не изменились — строка не дописана")
            return False
    pd.DataFrame([row], columns=HISTORY_COLS).to_csv(path, mode="a", header=not path.exists(),
                                                     index=False, encoding="utf-8-sig")   # только дописываем
    log.append(f"INFO  progress_history.csv: дописана строка (всего {row['всего']}%)")
    return True


# ───────────────────────────── картинка ─────────────────────────────

def render_tracker(df, out_path, scale=0.7):
    df = numeric(df)

    columns, chapters, steps = [], [], []
    for stage in ORDER:
        start = len(columns)
        for sub, grp in df[df["stage"] == stage].groupby("subsection", sort=False):
            chapters.append((len(columns), sub))
            columns.append(grp.to_dict("records"))
        if len(columns) == start:
            continue                                           # пустой этап не рисуем
        steps.append((start, len(columns) - 1, stage))         # первый и последний столбец этапа
        columns.append(None)                                   # отступ между этапами
    columns.pop()

    n_cols, n_rows = len(columns), max(len(c) for c in columns if c)
    legend_y = n_rows + 1.3                                    # строка легенды — под контейнерами
    s = scale
    x_max = max(n_cols + 3.2, draw_legend(None, x=0, y=legend_y, s=s))   # справа место под градусник (и легенду, если она длиннее)
    fig, ax = plt.subplots(figsize=((x_max + 0.4) * 0.33 * s + 0.3, (n_rows + 12) * 0.33 * s))

    for first, last, stage in steps:                           # контейнеры этапов + названия
        ax.add_patch(FancyBboxPatch((first - 0.15, TITLE_Y - 1.0), last - first + 1.3, n_rows - TITLE_Y + 1.1,
                                    boxstyle="round,pad=0,rounding_size=0.5",
                                    facecolor="#f6f8fa", edgecolor="#d0d7de", linewidth=1, zorder=0))
        ax.text((first + last + 1) / 2, TITLE_Y, stage.upper(), ha="center", va="center",
                fontsize=10 * s, fontweight="bold", color="#24292f")

    for x, name in chapters:                                   # подписи разделов — вплотную над своими кубиками
        ax.text(x + 0.5, LABEL_TOP, name, rotation=90, ha="center", va="bottom",
                fontsize=8.5 * s, color="#57606a")

    for x, col in enumerate(columns):                          # кубики со слоями
        for y, t in enumerate(col or []):
            draw_cube(ax, x, y, t["share"], undone=t["undone"], in_progress=t["status"] == "in_progress",
                      missing=t["status"] not in LEARNED, broken=t["test_run"] == 1, s=s)

    draw_legend(ax, x=0, y=legend_y, s=s)

    ax.set_xlim(-0.4, x_max)
    ax.set_ylim(legend_y + 1.0, TITLE_Y - 1.3)                 # ось перевёрнута: сверху этапы и подписи, ниже кубики и легенда
    ax.set_aspect("equal")
    ax.axis("off")

    progress = df["share"].mean()                              # общий прогресс по всему роадмапу (= overall_progress)
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


def draw_cube(ax, x, y, share, undone=0, in_progress=False, missing=False, broken=False, s=0.7):
    """Кубик темы: цвет (повторения) → штриховка (не сделано) → оранжевая рамка (in_progress) → значки «?» и «!»."""
    x0, y0, size = x + 0.08, y + 0.08, 0.84
    face = CMAP(float(share))                                  # float: CMAP(1) с целым — это индекс, а не 100%
    cube = FancyBboxPatch((x0, y0), size, size, boxstyle="round,pad=0,rounding_size=0.18",
                          facecolor=face, edgecolor="none", zorder=1)
    ax.add_patch(cube)

    if undone:                                                 # 1. тонкая диагональная решётка, 4 уровня густоты
        step = size / (undone + 1.5)                           # чем больше несделанного, тем чаще линии
        color = hatch_color(face)
        for c in np.arange(-size, size, step) + step / 2:      # линии под 45° в обе стороны → решётка «x»
            for sign in (1, -1):
                xs = np.array([x0, x0 + size])
                ys = (y0 + c + (xs - x0)) if sign > 0 else (y0 + size - c - (xs - x0))
                line, = ax.plot(xs, ys, color=color, linewidth=0.35 * s, zorder=2, solid_capstyle="butt")
                line.set_clip_path(cube)                       # обрезаем по скруглённому кубику

    if in_progress:                                            # 2. оранжевая рамка — тема в работе
        ax.add_patch(FancyBboxPatch((x0, y0), size, size, boxstyle="round,pad=0,rounding_size=0.18",
                                    facecolor="none", edgecolor=IN_PROGRESS_COLOR, linewidth=1.6 * s, zorder=3))
    if missing:                                                # 3. «?» в левом верхнем углу — нет статуса в шапке
        draw_badge(ax, x0 + 0.2, y0 + 0.2, "?", MISSING_COLOR, s)
    if broken:                                                 # 4. «!» в правом верхнем углу — тест сломан
        draw_badge(ax, x0 + size - 0.2, y0 + 0.2, "!", BROKEN_COLOR, s)


def draw_legend(ax, x, y, s):
    """Компактная легенда в одну строку: штриховка, рамка, «!», «?». Возвращает правый край.
    С ax=None ничего не рисует — только считает ширину (нужно заранее для размера картинки)."""
    def cube(px, **layers):
        if ax is not None:
            draw_cube(ax, px, y, layers.pop("share", 0), s=s, **layers)

    def label(px, text):
        if ax is not None:
            ax.text(px, y + 0.5, text, ha="left", va="center", fontsize=8 * s, color="#57606a")
        return px + len(text) * 0.25 + 1.2                     # грубая оценка ширины текста в клетках + отступ

    for k in range(1, 5):                                      # четыре уровня штриховки на сером фоне
        cube(x + k - 1, undone=k)
    x = label(x + 4.2, "не сделано обязательных задач этапа: 1–4")
    cube(x, undone=3, in_progress=True)
    x = label(x + 1.2, "in_progress")
    cube(x, share=0.5, broken=True)
    x = label(x + 1.2, "тест сломан (test_run = 1)")
    cube(x, share=0.5, missing=True)
    return label(x + 1.2, "нет статуса в шапке")


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


# ───────────────────────────── радар ─────────────────────────────
# Согласованный вид (из radar.py) — внешний вид не менять.

GREEN, GREEN_FILL = "#216e39", "#40c463"                         # зелёная гамма, как у кубиков на карте
GRID, TEXT, MUTED = "#d0d7de", "#24292f", "#57606a"
STEPS = (0.05, 0.10, 0.25, 0.50, 1.00)                           # возможные «потолки» шкалы: 5%, 10%, 25%, 50%, 100%



def pick_scale(vals):
    """Самый маленький потолок, в который влезает максимум с небольшим запасом."""
    top = max(vals.max(), 1e-9) * 1.1                            # *1.1 — запас, чтобы точка не лежала на самом краю
    return next(s for s in STEPS if s >= top)                    # первый подходящий; если почти 100% — будет 1.0


def render_radar(values, out_path, total=None, size=3.2):
    """values — словарь {этап: доля готовности от 0 до 1}; total — общий прогресс (если None — среднее по этапам)."""
    names = list(values)
    vals = np.array([values[n] for n in names])
    n = len(names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)        # угол каждого луча
    scale = pick_scale(vals)                                     # текущий потолок шкалы
    r_vals = vals / scale                                        # растягиваем значения: потолок = внешнее кольцо

    fig = plt.figure(figsize=(size * 2.1, size))
    ax = fig.add_axes([0.02, 0.06, 0.48, 0.88], polar=True)      # радар — левая половина картинки
    ax.set_theta_offset(np.pi / 2)                               # первый луч смотрит вверх
    ax.set_theta_direction(-1)                                   # лучи идут по часовой стрелке

    for r in (0.25, 0.5, 0.75, 1.0):                             # кольца-уровни без подписей
        ax.plot(np.linspace(0, 2 * np.pi, 200), [r] * 200, color=GRID, lw=0.8)
    for a in angles:                                             # сами лучи
        ax.plot([a, a], [0, 1], color=GRID, lw=0.8)

    closed_a = np.append(angles, angles[0])                      # замыкаем многоугольник на первую точку
    closed_v = np.append(r_vals, r_vals[0])
    ax.fill(closed_a, closed_v, color=GREEN_FILL, alpha=0.35)
    ax.plot(closed_a, closed_v, color=GREEN, lw=1.5)
    ax.scatter(angles, r_vals, color=GREEN, s=12, zorder=3)
    ax.text(np.radians(22), 1.0, f"{scale:.0%}", fontsize=6.5, color=MUTED,   # подпись: сколько сейчас значит внешнее кольцо
            ha="left", va="bottom")

    for a, name in zip(angles, names):                           # подпись поперёк луча
        deg = np.degrees(a)                                      # 0° — верх, дальше по часовой
        rot = -deg                                               # повернуть текст перпендикулярно лучу
        if 90 < deg < 270:                                       # нижняя половина — перевернуть, чтобы не читать вверх ногами
            rot += 180
        ax.text(a, 1.16, name, rotation=rot, rotation_mode="anchor",
                ha="center", va="center", fontsize=8, color=TEXT)

    ax.set_ylim(0, 1.25)
    ax.axis("off")                                               # убираем стандартную сетку и подписи matplotlib

    tab = fig.add_axes([0.56, 0.12, 0.42, 0.76]); tab.axis("off")   # таблица — правая половина
    rows = [[name, f"{v:.0%}"] for name, v in zip(names, vals)]
    rows.append(["всего", f"{(vals.mean() if total is None else total):.0%}"])   # «всего» = общий прогресс, как на карте
    t = tab.table(cellText=rows, colLabels=["этап", "готовность"], loc="center",
                  colWidths=[0.65, 0.35], cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.35)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID); cell.set_linewidth(0.6)
        if c == 1: cell.set_text_props(ha="right")               # проценты выровнены вправо
        if r == 0: cell.set_text_props(weight="bold", color=TEXT); cell.set_facecolor("#f6f8fa")
        elif r == len(rows): cell.set_text_props(weight="bold")  # строка «всего» жирным
        else: cell.set_text_props(color=MUTED if c == 0 else TEXT)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def overall_progress(df):
    """Общий процент «насколько знаю сейчас»: среднее score / 10 по всем темам. Тот же, что в заголовке карты."""
    return numeric(df)["share"].mean()


def radar_values(df):
    """{этап: насколько знаю сейчас, 0..1} в порядке этапов, как на карте.
    Знание темы = score / 10: засчитанные повторы минус штраф за простой — та же мера, что цвет кубиков."""
    d = numeric(df)
    return {stage: float(d.loc[d["stage"] == stage, "share"].mean()) if (d["stage"] == stage).any() else 0.0
            for stage in ORDER}


# ───────────────────────────── README-визитка ─────────────────────────────

START, END = "<!-- progress:start -->", "<!-- progress:end -->"
TITLES = {"eda": "EDA", "preprocessing": "Preprocessing", "model": "Model", "activations": "Activations",
          "losses": "Losses", "optimizers": "Optimizers", "metrics": "Metrics"}


def build_readme(df):
    """Витрина для работодателя: шапка, как читать, карта, радар, что в работе, темы по этапам. Всё — из таблицы."""
    in_work = df.loc[df["status"] == "in_progress", "slug"].tolist()
    out = ["<!-- Файл собирается сам: scripts/update_progress.py. Руками не править. -->",
           "# ROADMAP-ML", "",
           "**Моя личная библиотека ML — техкарта знаний и навыков**", "",
           "ML / Data Science разработчик", "",
           "## Как читать карту", "",
           "- Каждый квадрат — тема.",
           "- Ярче зелёный — увереннее владею: больше свежих повторений.",
           "- Меньше штриховки — полнее проработана: теория, конспект, эксперимент, тест.",
           "- Оранжевая рамка — в работе сейчас.", "",
           "## Карта прогресса", "", START, "![ROADMAP-ML progress](assets/progress.png)", END, "",
           "## Радар навыков", "", "![Радар навыков](assets/radar.png)", "",
           "_Масштаб радара подстраивается под текущий уровень — точные значения в таблице справа._", "",
           "## Сейчас в работе", "", " · ".join(in_work) if in_work else "—", "",
           "## Темы", ""]
    for stage in ORDER:
        g = df[df["stage"] == stage]
        if g.empty:
            continue
        out += [f"<details><summary><b>{TITLES.get(stage, stage)}</b> — {len(g)}</summary>", "",
                "| Подраздел | Темы |", "|---|---|"]
        for sub, sg in g.groupby("subsection", sort=False):
            out.append(f"| {sub} | {', '.join(sg['slug'])} |")
        out += ["", "</details>", ""]
    return "\n".join(out).rstrip() + "\n"


def problems(df, results):
    """Для ноутбука: упавшие тесты [(тест, файл)] и файлы без корректного статуса."""
    failed = [case for res in results.values() for case in res["failed"]]
    no_status = df.loc[~df["status"].isin(LEARNED), "path"].tolist()
    return {"failed_tests": failed, "no_status": no_status}


# ───────────────────────────── запуск ─────────────────────────────

def main(root=ROOT, verbose=True, skip_tests=False):
    """Полный цикл. Возвращает таблицу, строки лога и проблемы (удобно для ноутбука).
    skip_tests=True — не гонять pytest: повторы и test_run берутся из прошлой таблицы."""
    root = Path(root).resolve()
    csv_path, log_path = root / "roadmap_tracker.csv", root / "logs" / "progress.log"
    old = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str, keep_default_na=False) \
        if csv_path.exists() else pd.DataFrame(columns=COLUMNS)   # старая таблица нужна ради порядка строк и --skip-tests
    log = []
    results = run_tests(root, log, run=not skip_tests)
    df = build_table(root, old, log, results, ran_tests=not skip_tests)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    append_history(root, df, log)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path.write_text(f"# update_progress {stamp}\n" + "\n".join(log) + "\n", encoding="utf-8")   # перезапись при каждом запуске

    render_tracker(df, root / "assets" / "progress.png")
    render_radar(radar_values(df), root / "assets" / "radar.png", total=overall_progress(df))   # «всего» = как в заголовке карты
    (root / "README.md").write_text(build_readme(df), encoding="utf-8")

    if verbose:
        for line in log:
            if not line.startswith("INFO"):
                print(line)
        print(stage_summary(df).to_string(index=False))
        print(f"Готово: roadmap_tracker.csv, progress_history.csv, logs/progress.log, assets/progress.png, "
              f"assets/radar.png, README.md (тем: {len(df)}, предупреждений: {sum(l.startswith('WARN') for l in log)})")
    return df, log, problems(df, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="корень проекта (по умолчанию — репозиторий)")
    parser.add_argument("--skip-tests", action="store_true",
                        help="не гонять pytest: повторы и test_run взять из прошлой таблицы")
    args = parser.parse_args()
    TABLE, LOG, PROBLEMS = main(args.root, skip_tests=args.skip_tests)   # после %run в ноутбуке эти имена доступны
