"""Собирает трекер ROADMAP-ML из архитектуры src/, рисует картинку прогресса, пишет историю и README-отчёт.

Запуск из корня репозитория:  python scripts/update_progress.py [--pytest]
Структура: src/<этап>/<подраздел>/<метод>.py  или  src/<этап>/<подраздел>/<метод>/<метод>.py
"""
import argparse
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
ORDER = ["eda", "preprocessing", "model", "activations", "losses", "optimizers", "metrics"]   # этапы = папки src/
FULL = 4                                                       # сколько повторений = 100% «осело»
CMAP = LinearSegmentedColormap.from_list(                      # плавная шкала: серый → тёмно-зелёный
    "progress", ["#ebedf0", "#9be9a8", "#40c463", "#30a14e", "#216e39"])
LABEL_TOP, TITLE_Y = -0.35, -8.6                               # где подписи разделов и названия этапов (над кубиками)

COLUMNS = ["stage", "subsection", "slug", "path", "description", "status", "learned",
           "repeated", "readme", "experiment", "test_create", "test_run"]
MANUAL = ["repeated", "test_run"]                              # ручные поля: при пересборке переносятся по path
LEARNED = {"todo": 0, "in_progress": 0.5, "ready": 1, "готов": 1, "done": 1}   # статус → learned
SKIP_DIRS = {"attempts", "__pycache__"}                        # черновики и кэш — не методы
HISTORY_COLS = ["datetime", "всего"] + ORDER                   # progress_history.csv

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


# ───────────────────────────── таблица ─────────────────────────────

def build_table(root, old, log, test_results=None):
    """Пересобирает таблицу из src/. Ручные поля (repeated, test_run) берёт из старой таблицы по path."""
    manual = old.set_index("path")[[c for c in MANUAL if c in old.columns]].to_dict("index") \
        if "path" in old.columns else {}
    order = {p: i for i, p in enumerate(old["path"])} if "path" in old.columns else {}   # прежний порядок строк
    tests = {p.name for p in (root / "tests").rglob("test_*.py")} if (root / "tests").is_dir() else set()

    rows = []
    for m in scan_src(root, log):
        if m.status not in LEARNED:
            log.append(f"WARN  {m.path}: " + (f"неизвестный статус «{m.status}» (ожидается todo / in_progress / ready)"
                                              if m.status else "статус отсутствует (нет строки `# status:` в шапке)"))
        kept = manual.get(m.path, {})
        exp_dir = root / "experiments" / m.slug
        test_run = kept.get("test_run", 0) if test_results is None else test_results.get(f"test_{m.slug}.py", 0)
        rows.append({"stage": m.stage, "subsection": m.subsection, "slug": m.slug, "path": m.path,
                     "description": m.description, "status": m.status,
                     "learned": LEARNED.get(m.status, 0),       # learned — только из статуса
                     "repeated": kept.get("repeated", 0) or 0,
                     "readme": readme_has_text(m.method_dir),
                     "experiment": int(exp_dir.is_dir() and any(exp_dir.glob("*.ipynb"))),
                     "test_create": int(f"test_{m.slug}.py" in tests),
                     "test_run": test_run or 0})
    df = pd.DataFrame(rows, columns=COLUMNS).astype(object)
    df["learned"] = [f"{v:g}" for v in df["learned"]]          # 0 / 0.5 / 1, а не 0.0

    gone = set(manual) - set(df["path"])                       # файл удалён → строка пропала (сообщаем)
    for path in sorted(gone):
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


# ───────────────────────────── сводка (для README, истории и ноутбука) ─────────────────────────────

def numeric(df):
    """Копия таблицы с числами + undone (сколько из четырёх задач не сделано) и share (доля повторений)."""
    df = df.copy()
    for col in ["learned", "repeated", "readme", "experiment", "test_create", "test_run"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["undone"] = ((df["learned"] < 1).astype(int) + (df["readme"] == 0) +  # сколько из четырёх задач не сделано
                    (df["experiment"] == 0) + (df["test_create"] == 0))
    df["share"] = df["repeated"].clip(upper=FULL) / FULL                      # повторения → доля от 0 до 1
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
        return px + len(text) * 0.22 + 1.2                     # грубая оценка ширины текста в клетках + отступ

    for k in range(1, 5):                                      # четыре уровня штриховки на сером фоне
        cube(x + k - 1, undone=k)
    x = label(x + 4.2, "не сделано 1–4 из: learned, readme, experiment, test")
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


# ───────────────────────────── README-отчёт ─────────────────────────────

START, END = "<!-- progress:start -->", "<!-- progress:end -->"


def cell(text):
    """Текст для ячейки markdown-таблицы: без переносов и с экранированным «|»."""
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def file_link(path):
    return f"[`{Path(path).name}`]({path})" if path else "—"


def topics_table(g):
    """Таблица тем: тема · этап / подраздел · файл · где применяется."""
    if g.empty:
        return "_Пока пусто._\n"
    lines = ["| Тема | Этап / подраздел | Файл | Где применяется |", "|---|---|---|---|"]
    for _, t in g.iterrows():
        lines.append(f"| {t['slug']} | {t['stage']} / {t['subsection']} | {file_link(t['path'])} | "
                     f"{cell(t['description']) or '—'} |")
    return "\n".join(lines) + "\n"


def build_readme(df):
    """Личный отчёт: картинка, проценты, изучено / в работе / планы, проблемы. Всё — из таблицы."""
    d = numeric(df)
    summary = stage_summary(df)
    total = summary.iloc[-1]
    out = ["# ROADMAP-ML", "",
           "Мой личный трекер пути в ML: что уже изучено, что в работе, что впереди и где сейчас проблемы.",
           "Этот файл собирается сам из `src/` скриптом `scripts/update_progress.py` — руками не править.",
           "", START, "![ROADMAP-ML progress](assets/progress.png)", END, "",
           "## Прогресс", "",
           f"**Готовность: {total['готовность, %']:g}%** · ready {total['ready']} из {total['тем']} тем · "
           f"in_progress {total['in_progress']} · todo {total['todo']} · "
           f"закреплено повторениями: {total['повторения, %']:g}%", "",
           "| Этап | Тем | ready | in_progress | todo | Готовность | Повторения |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in summary.iterrows():
        name = f"**{r['этап']}**" if r["этап"] == "всего" else r["этап"]
        out.append(f"| {name} | {r['тем']} | {r['ready']} | {r['in_progress']} | {r['todo']} | "
                   f"{r['готовность, %']:g}% | {r['повторения, %']:g}% |")
    out += ["", "_Готовность — среднее `learned` (ready = 1, in_progress = 0.5, todo = 0). "
                "Повторения — `repeated` / 4, по ним красятся кубики на картинке. "
                "Динамика — в `progress_history.csv`._", ""]

    out += ["## Изучено", "", topics_table(d[d["learned"] == 1]),
            "## В работе", "", topics_table(d[d["status"] == "in_progress"]),
            "## Планы", "", "_Статус `todo` — ещё предстоит выучить._", ""]
    plans = d[d["status"] == "todo"]
    for stage in ORDER:
        g = plans[plans["stage"] == stage]
        if g.empty:
            continue
        out.append(f"<details><summary><b>{stage}</b> — {len(g)}</summary>\n")
        for sub, sg in g.groupby("subsection", sort=False):
            out.append(f"- **{sub}**: " + ", ".join(sg["slug"]))
        out += ["", "</details>", ""]

    no_status = d[~d["status"].isin(LEARNED)]
    broken = d[d["test_run"] == 1]
    out += ["## Проблемы", ""]
    if no_status.empty and broken.empty:
        out += ["_Проблем нет._", ""]
    if not broken.empty:
        out += [f"**`!` Сломан тест — {len(broken)}**", ""]
        out += [f"- {t['slug']} — {file_link(t['path'])}" for _, t in broken.iterrows()] + [""]
    if not no_status.empty:
        out += [f"**`?` Нет или неверный `status` в шапке — {len(no_status)}**", ""]
        out += [f"- {t['slug']} — {file_link(t['path'])}" for _, t in no_status.iterrows()] + [""]
    return "\n".join(out).rstrip() + "\n"


# ───────────────────────────── запуск ─────────────────────────────

def main(root=ROOT, verbose=True, pytest=False):
    """Полный цикл. Возвращает таблицу и строки лога (удобно для ноутбука).
    pytest=True — прогнать тесты и проставить test_run по факту, иначе test_run берётся из таблицы."""
    root = Path(root).resolve()
    csv_path, log_path = root / "roadmap_tracker.csv", root / "logs" / "progress.log"
    old = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str, keep_default_na=False) \
        if csv_path.exists() else pd.DataFrame(columns=COLUMNS)   # старая таблица нужна только ради ручных полей
    log = []
    test_results = run_pytest(root, log) if pytest else None
    df = build_table(root, old, log, test_results)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    append_history(root, df, log)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path.write_text(f"# update_progress {stamp}\n" + "\n".join(log) + "\n", encoding="utf-8")   # перезапись при каждом запуске

    render_tracker(df, root / "assets" / "progress.png")
    (root / "README.md").write_text(build_readme(df), encoding="utf-8")

    if verbose:
        for line in log:
            if not line.startswith("INFO"):
                print(line)
        print(stage_summary(df).to_string(index=False))
        print(f"Готово: roadmap_tracker.csv, progress_history.csv, logs/progress.log, assets/progress.png, README.md "
              f"(тем: {len(df)}, предупреждений: {sum(l.startswith('WARN') for l in log)})")
    return df, log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="корень проекта (по умолчанию — репозиторий)")
    parser.add_argument("--pytest", action="store_true", help="прогнать pytest и проставить test_run автоматически")
    args = parser.parse_args()
    main(args.root, pytest=args.pytest)
