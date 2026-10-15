"""Графики панели ROADMAP-ML. Каждый — в рамке, как контейнеры этапов на карте.

В ноутбуке:
    import charts
    charts.show_all(df, problems, ROOT)      # df и problems — из update_progress.main()

Порядок: итог → карта (крупно, 2 части) → этапы × задачи → сейчас в работе → радар → повторы → упали тесты → нет статуса.
"""
import calendar
import textwrap
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import update_progress as up

# цвета — как на карте
INK, MUTED, FAINT, GRID, PANEL = "#24292f", "#57606a", "#8c959f", "#d0d7de", "#f6f8fa"
STATE_COLOR = {"закрыто": "#30a14e", "в работе": "#fb8500", "не начато": "#d0d7de", "без статуса": "#8250df"}
BROKEN = "#cf222e"                                             # «!» — упал тест
MONTHS = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]
plt.rcParams["font.family"] = "DejaVu Sans"                   # один шрифт везде (иначе жирный текст «прыгает» в засечки)
WIDTH = 12                                                     # ширина всех рамок, дюймы — чтобы шли ровной колонкой


# ───────────────────────────── общее: рамка и данные ─────────────────────────────

def frame(title, height, note=None):
    """Фигура с рамкой как у этапа на карте: скруглённая серая подложка + заголовок. Возвращает (fig, область внутри)."""
    fig = plt.figure(figsize=(WIDTH, height), facecolor="white")
    fig.patches.append(FancyBboxPatch((0.006, 0.02), 0.988, 0.96, boxstyle="round,pad=0,rounding_size=0.012",
                                      mutation_aspect=WIDTH / height,   # скругление одинаковое по ширине и высоте
                                      transform=fig.transFigure, facecolor=PANEL, edgecolor=GRID, linewidth=1,
                                      zorder=-10))             # подложка — самый нижний слой, под графиками
    fig.text(0.025, 1 - 0.32 / height, title.upper(), fontsize=11, fontweight="bold", color=INK, va="center")
    if note:
        fig.text(0.025, 1 - 0.58 / height, note, fontsize=9, color=MUTED, va="center")
    top = 1 - (0.82 if note else 0.55) / height
    return fig, (0.025, 0.06, 0.95, top - 0.06)


def card(ax, x, y, w, h, color="white", radius_px=12):
    """Белая карточка со скруглением ~12 px внутри рамки (координаты осей 0..1)."""
    bbox = ax.get_window_extent()
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius_px / bbox.width}",
                                mutation_aspect=bbox.width / bbox.height,   # радиус одинаковый по x и y
                                transform=ax.transAxes, facecolor=color, edgecolor=GRID, linewidth=0.8))


def state_of(status):
    return {"ready": "закрыто", "готов": "закрыто", "done": "закрыто", "in_progress": "в работе",
            "todo": "не начато"}.get(status, "без статуса")


def prepare(df):
    d = up.numeric(df)
    d["state"] = d["status"].map(state_of)
    return d


# ───────────────────────────── 1. итог ─────────────────────────────

def chart_summary(d, df):
    """Крупные числа в карточках."""
    tiles = [("тем всего", len(d), INK), ("закрыто", (d.state == "закрыто").sum(), STATE_COLOR["закрыто"]),
             ("в работе", (d.state == "в работе").sum(), STATE_COLOR["в работе"]),
             ("не начато", (d.state == "не начато").sum(), MUTED),
             ("нет статуса ?", (d.state == "без статуса").sum(), STATE_COLOR["без статуса"]),
             ("упал тест !", (d.test_run == 1).sum(), BROKEN),
             ("знаю сейчас", f"{up.overall_progress(df):.0%}", INK)]
    fig, rect = frame("Итог", 2.1)
    ax = fig.add_axes(rect); ax.axis("off")
    n, gap = len(tiles), 0.012
    w = (1 - gap * (n - 1)) / n
    for i, (label, value, color) in enumerate(tiles):
        x = i * (w + gap)
        card(ax, x, 0.04, w, 0.92)
        ax.text(x + w / 2, 0.58, str(value), ha="center", va="center", fontsize=22, fontweight="bold",
                color=color, transform=ax.transAxes)
        ax.text(x + w / 2, 0.22, label, ha="center", va="center", fontsize=9.5, color=MUTED, transform=ax.transAxes)
    return fig


# ───────────────────────────── 2. матрица «этап × задача» ─────────────────────────────

TASKS = [("статус ready", "ready"), ("README", "readme"), ("тест", "test"), ("повторы", "repeat")]


def task_done(d):
    """Сделана ли задача в каждой теме: статус ready, README расписан, тест есть, есть живые повторы (score > 0)."""
    return {"ready": d.learned >= 1, "readme": d.readme == 1, "test": d.test_create == 1, "repeat": d.score > 0}


def chart_matrix(d):
    """Тепловая карта (heatmap): строка — этап, столбец — задача. Цвет — доля тем, где сделано (шкала как у карты)."""
    stages = [s for s in up.ORDER if (d.stage == s).any()]
    done = task_done(d)
    fig, rect = frame("Этапы × задачи", 0.62 * len(stages) + 1.9,
                      note="В клетке — сколько тем этапа закрыли задачу. Чем зеленее, тем больше сделано. «не нужно» — задача этапу не обязательна.")
    x0, y0, w, h = rect
    ax = fig.add_axes((x0, y0, w, h)); ax.axis("off")
    n_cols, label_w = len(TASKS), 1.6                          # слева колонка с названиями этапов
    ax.set_xlim(0, label_w + n_cols + 1.1); ax.set_ylim(len(stages) + 0.9, 0)
    for j, (name, _) in enumerate(TASKS):                      # шапка столбцов
        ax.text(label_w + j + 0.5, 0.45, name, ha="center", va="center", fontsize=10, fontweight="bold", color=INK)
    ax.text(label_w + n_cols + 0.55, 0.45, "тем", ha="center", va="center", fontsize=10, fontweight="bold", color=INK)
    for i, stage in enumerate(stages):
        y = i + 1
        m = d.stage == stage
        total = int(m.sum())
        need = set(up.REQUIRED.get(stage, up.TASKS)) | {"repeat"}   # повторы считаем у всех этапов
        ax.text(label_w - 0.15, y + 0.4, stage, ha="right", va="center", fontsize=10.5, color=INK)
        for j, (_, key) in enumerate(TASKS):
            x = label_w + j
            if key not in need:                                # задача этапу не нужна — серая клетка
                ax.add_patch(FancyBboxPatch((x + 0.04, y), 0.92, 0.8, boxstyle="round,pad=0,rounding_size=0.08",
                                            facecolor="#eef1f4", edgecolor="none"))
                ax.text(x + 0.5, y + 0.4, "не нужно", ha="center", va="center", fontsize=8.5, color=FAINT)
                continue
            k = int((done[key] & m).sum())
            share = k / total if total else 0
            face = up.CMAP(0.12 + 0.88 * share) if k else "#ebedf0"   # пусто — серый, как незакрашенный кубик
            ax.add_patch(FancyBboxPatch((x + 0.04, y), 0.92, 0.8, boxstyle="round,pad=0,rounding_size=0.08",
                                        facecolor=face, edgecolor="none"))
            ink = "white" if share > 0.55 else INK
            ax.text(x + 0.5, y + 0.3, f"{k}/{total}", ha="center", va="center", fontsize=10.5, fontweight="bold", color=ink)
            ax.text(x + 0.5, y + 0.6, f"{share:.0%}", ha="center", va="center", fontsize=8.5,
                    color="white" if share > 0.55 else MUTED)
        ax.text(label_w + n_cols + 0.55, y + 0.4, str(total), ha="center", va="center", fontsize=10.5, color=MUTED)
    return fig


# ───────────────────────────── 3. сейчас в работе ─────────────────────────────

def chart_focus(d):
    """Чек-лист по темам in_progress: что уже есть (✓), чего не хватает (✗), повторы и флаги «!» / «?»."""
    work = d[d.status == "in_progress"]
    done = task_done(d)
    rows = max(len(work), 1)
    fig, rect = frame("Сейчас в работе", 0.55 * rows + 1.75,
                      note="Что осталось сделать по темам со статусом in_progress. Зелёная плашка — сделано, серая — нет.")
    x0, y0, w, h = rect
    ax = fig.add_axes((x0, y0, w, h)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(rows + 0.9, 0)
    if work.empty:
        ax.text(5, 0.9, "Тем в работе нет — поставь status: in_progress в шапке темы, которую начинаешь",
                ha="center", va="center", fontsize=10.5, color=MUTED)
        return fig
    heads = [("тема", 0.05, "left"), ("статус ready", 3.55, "center"), ("README", 4.75, "center"),
             ("тест", 5.95, "center"), ("повторы", 7.55, "center"), ("флаги", 9.3, "center")]
    for name, x, ha in heads:
        ax.text(x, 0.45, name, ha=ha, va="center", fontsize=9.5, fontweight="bold", color=INK)
    for i, (idx, t) in enumerate(work.iterrows()):
        y = i + 1
        ax.text(0.05, y + 0.28, t.slug, va="center", fontsize=10.5, color=INK)
        ax.text(0.05, y + 0.62, f"{t.stage} · {t.subsection}", va="center", fontsize=8.5, color=MUTED)
        need = up.REQUIRED.get(t.stage, up.TASKS)
        for key, x in (("ready", 3.55), ("readme", 4.75), ("test", 5.95)):   # плашки ✓ / ✗ / «не нужно»
            if key not in need:
                face, label, ink = "#eef1f4", "не нужно", FAINT
            elif done[key][idx]:
                face, label, ink = STATE_COLOR["закрыто"], "✓ есть", "white"
            else:
                face, label, ink = "#ebedf0", "✗ нет", MUTED
            ax.add_patch(FancyBboxPatch((x - 0.5, y + 0.12), 1.0, 0.6, boxstyle="round,pad=0,rounding_size=0.12",
                                        mutation_aspect=0.5, facecolor=face, edgecolor="none"))
            ax.text(x, y + 0.42, label, ha="center", va="center", fontsize=8.5, color=ink)
        score = int(t.score)                                   # полоска повторов 0..10
        ax.add_patch(FancyBboxPatch((6.85, y + 0.32), 1.4, 0.2, boxstyle="round,pad=0,rounding_size=0.08",
                                    mutation_aspect=0.5, facecolor="#ebedf0", edgecolor="none"))
        if score:
            ax.add_patch(FancyBboxPatch((6.85, y + 0.32), 1.4 * score / up.FULL, 0.2,
                                        boxstyle="round,pad=0,rounding_size=0.08", mutation_aspect=0.5,
                                        facecolor=up.CMAP(0.25 + 0.75 * score / up.FULL), edgecolor="none"))
        ax.text(8.35, y + 0.42, f"{score}/{up.FULL}", va="center", fontsize=9, color=INK)
        flags = [("!", BROKEN)] * int(t.test_run == 1) + [("?", STATE_COLOR["без статуса"])] * int(t.state == "без статуса")
        for k, (sign, color) in enumerate(flags or [("—", None)]):
            if color is None:
                ax.text(9.3, y + 0.42, "—", ha="center", va="center", fontsize=10, color=FAINT)
            else:
                ax.scatter(9.15 + k * 0.3, y + 0.42, s=170, color=color, zorder=3)
                ax.text(9.15 + k * 0.3, y + 0.42, sign, ha="center", va="center", fontsize=9, fontweight="bold",
                        color="white", zorder=4)
        if i < len(work) - 1:
            ax.plot([0.05, 9.95], [y + 0.88, y + 0.88], color=GRID, lw=0.6)
    return fig


# ───────────────────────────── 4. карта (крупно, в две части) ─────────────────────────────

MAP_PARTS = [["eda", "preprocessing", "model"], ["layers", "activations", "losses", "optimizers", "metrics"]]


def chart_map(df):
    """Та же карта, что в README (кубики, штриховка, рамки in_progress, «!» и «?»), но крупно и в две части."""
    import io
    total = up.overall_progress(df)
    images = []
    for i, stages in enumerate(MAP_PARTS, 1):
        buf = io.BytesIO()
        up.render_tracker(df, buf, scale=1.0, stages=stages,
                          title=f"Карта · часть {i} из {len(MAP_PARTS)} · прогресс {total:.0%}")
        images.append(buf.getvalue())
    return images                                              # PNG-байты: карта рисует свои рамки сама


# ───────────────────────────── 5. радар ─────────────────────────────

def chart_radar(df):
    """Радар с нуля, в рамке: слева роза ветров (как согласовано), справа — список этапов с полосками."""
    values = up.radar_values(df)                               # {этап: насколько знаю сейчас, 0..1}
    names, vals = list(values), np.array(list(values.values()))
    scale = up.pick_scale(vals)                                # автомасштаб: 5 / 10 / 25 / 50 / 100%
    fig, rect = frame("Радар навыков · знаю сейчас", 5.0,
                      note="Масштаб подстраивается под текущий уровень — точные значения в списке справа.")
    x0, y0, w, h = rect

    ax = fig.add_axes((x0, y0, w * 0.45, h), polar=True)       # роза ветров — левая часть
    ax.set_facecolor("none")                                   # прозрачный фон — сливается с рамкой
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
    r_vals = vals / scale
    for r in (0.25, 0.5, 0.75, 1.0):                           # кольца без подписей
        ax.plot(np.linspace(0, 2 * np.pi, 200), [r] * 200, color=up.GRID, lw=0.8)
    for a in angles:
        ax.plot([a, a], [0, 1], color=up.GRID, lw=0.8)
    ca, cv = np.append(angles, angles[0]), np.append(r_vals, r_vals[0])
    ax.fill(ca, cv, color=up.GREEN_FILL, alpha=0.35)
    ax.plot(ca, cv, color=up.GREEN, lw=1.5)
    ax.scatter(angles, r_vals, color=up.GREEN, s=14, zorder=3)
    ax.text(np.radians(22), 1.0, f"{scale:.0%}", fontsize=7.5, color=MUTED, ha="left", va="bottom")
    for a, name in zip(angles, names):                         # подписи поперёк луча, нижние перевёрнуты
        deg = np.degrees(a)
        rot = -deg + (180 if 90 < deg < 270 else 0)
        ax.text(a, 1.17, name, rotation=rot, rotation_mode="anchor", ha="center", va="center", fontsize=9, color=INK)
    ax.set_ylim(0, 1.28); ax.axis("off")

    side = fig.add_axes((x0 + w * 0.5, y0, w * 0.5, h)); side.axis("off")   # список: этап · полоска · процент
    card(side, 0, 0, 1, 1)
    rows = names + ["всего"]
    step = 0.86 / len(rows)
    for i, name in enumerate(rows):
        v = up.overall_progress(df) if name == "всего" else values[name]
        y = 0.92 - i * step
        bold = name == "всего"
        if bold:
            side.plot([0.05, 0.95], [y + step / 2, y + step / 2], color=GRID, lw=0.8, transform=side.transAxes)
        side.text(0.06, y, name, fontsize=10, color=INK, fontweight="bold" if bold else "normal",
                  va="center", transform=side.transAxes)
        side.add_patch(FancyBboxPatch((0.36, y - 0.018), 0.44, 0.036, boxstyle="round,pad=0,rounding_size=0.018",
                                      mutation_aspect=0.6, transform=side.transAxes, facecolor="#ebedf0",
                                      edgecolor="none"))                            # дорожка 0..потолок шкалы
        if v > 0:                                              # минимум 1.5% ширины — чтобы крошечное значение было видно
            side.add_patch(FancyBboxPatch((0.36, y - 0.018), max(0.44 * min(v / scale, 1), 0.015), 0.036,
                                          boxstyle="round,pad=0,rounding_size=0.018", mutation_aspect=0.6,
                                          transform=side.transAxes, facecolor=up.GREEN_FILL, edgecolor="none"))
        side.text(0.94, y, f"{v * 100:.1f}%", fontsize=10, color=INK, ha="right", va="center",
                  fontweight="bold" if bold else "normal", transform=side.transAxes)
    side.text(0.36, 0.035, f"полоска: 0 → {scale:.0%} (как внешнее кольцо)", fontsize=8, color=MUTED,
              transform=side.transAxes)
    return fig


# ───────────────────────────── 6. повторы ─────────────────────────────



def add_months(day, n):
    """Дата через n месяцев (31 янв + 1 мес → 28/29 фев)."""
    y, m = divmod(day.month - 1 + n, 12)
    y, m = day.year + y, m + 1
    return date(y, m, min(day.day, calendar.monthrange(y, m)[1]))


def score_at(dates, day):
    """Знание темы в день day: засчитанные к этому дню повторы минус полные месяцы с последнего (0..10)."""
    past = [x for x in dates if x <= day]
    if not past:
        return 0
    return max(0, min(up.FULL, len(past) - up.full_months(max(past), day)))


def repeat_dates(d, problems, root):
    """{путь эталона: (slug, этап, [даты засчитанных попыток по git])}."""
    info = {p: (s, st) for p, s, st in zip(d.path, d.slug, d.stage)}
    out = {}
    for path, stems in problems.get("passed", {}).items():
        topic = (Path(root) / path).parent
        dates = sorted(up.git_date(root, topic / "attempts" / f"{stem}.py") for stem in stems)
        slug, stage = info.get(path, (topic.name, ""))
        out[path] = (slug, stage, dates)
    return out


def inch_axes(fig, rect):
    """Оси в дюймах внутри рамки: (ax, ширина, высота). Ось y сверху вниз — удобно раскладывать строки."""
    x0, y0, w, h = rect
    W, H = WIDTH * w, fig.get_figheight() * h
    ax = fig.add_axes(rect); ax.axis("off")
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    return ax, W, H


def white_card(ax, x, y, w, h):
    """Белая карточка в дюймовых осях (скругление 0.12 дюйма)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor="white", edgecolor=GRID, linewidth=0.8))


def fmt_day(day, today):
    """15 окт · 15 окт 2025 (год — только если не текущий)."""
    return f"{day.day} {MONTHS[day.month - 1]}" + ("" if day.year == today.year else f" {day.year}")


def draw_calendar(cal, topics, today, weeks):
    """Календарь повторов как на GitHub: колонки — недели, строки — дни недели."""
    start = today - timedelta(days=today.weekday() + 7 * (weeks - 1))
    per_day = {}
    for _, _, dates in topics.values():
        for day in dates:
            per_day[day] = per_day.get(day, 0) + 1
    top = max(per_day.values(), default=1)
    for k in range((today - start).days + 1):
        day = start + timedelta(days=k)
        col, row = divmod(k, 7)
        n = per_day.get(day, 0)
        cal.add_patch(FancyBboxPatch((col + 0.1, 6 - row + 0.1), 0.8, 0.8, boxstyle="round,pad=0,rounding_size=0.18",
                                     facecolor=up.CMAP(0.3 + 0.7 * n / top) if n else "#ebedf0", edgecolor="none"))
        if day.day <= 7 and row == 0:
            cal.text(col + 0.1, 7.5, MONTHS[day.month - 1], fontsize=8, color=MUTED)
    for row, name in ((0, "пн"), (2, "ср"), (4, "пт")):
        cal.text(-0.4, 6 - row + 0.5, name, fontsize=7.5, color=MUTED, ha="right", va="center")
    cal.text(weeks + 0.8, 3.5, f"повторов: {sum(per_day.values())}\nза {weeks} недель", fontsize=10, color=INK, va="center")
    cal.set_xlim(-1.5, weeks + 7); cal.set_ylim(0, 8.2); cal.set_aspect("equal")


def chart_repeats(d, problems, root, weeks=26):
    """Календарь повторов + список: когда повторял, шкала 0..10, когда понизится на 1 и когда обнулится."""
    topics = repeat_dates(d, problems, root)
    today = date.today()
    rows = []
    for slug, stage, dates in topics.values():
        last, now = max(dates), score_at(dates, today)
        months = up.full_months(last, today)
        drop = add_months(last, months + 1)                    # следующий минус — когда «добьётся» очередной месяц
        zero = add_months(last, min(len(dates), up.FULL))      # без повторов знание дойдёт до 0
        rows.append((now == 0, (drop - today).days, slug, stage, len(dates), last, now, drop, zero))
    rows.sort()                                                # сначала самые срочные, обнулившиеся — в конце
    row_h = 0.62
    list_h = 0.75 + row_h * max(len(rows), 1)
    fig, rect = frame("Повторы", 2.3 + list_h + 1.45,
                      note="Повтор = attempts/attempt_NN.py, прошедший тест. Каждый полный месяц без нового повтора — минус 1.")
    x0, y0, w, h = rect
    H = fig.get_figheight()
    cal = fig.add_axes((x0, y0 + h - 2.0 / H, w, 2.0 / H)); cal.axis("off")
    draw_calendar(cal, topics, today, weeks)

    ax, W, _ = inch_axes(fig, (x0, y0 + 0.4 / H, w, (list_h + 0.1) / H))
    white_card(ax, 0, 0.05, W, list_h)
    if not rows:
        ax.text(W / 2, 0.05 + list_h / 2, "Засчитанных повторов пока нет — положи attempts/attempt_01.py в папку темы и закоммить",
                ha="center", va="center", fontsize=10.5, color=MUTED)
        return fig
    cols = {"тема": 0.3, "последний повтор": 3.1, "знаю сейчас (0–10)": 5.0, "понизится на 1": 7.9, "обнулится": 10.0}
    for name, x in cols.items():                               # шапка списка
        ax.text(x, 0.4, name, fontsize=9, color=MUTED, va="center")
    for i, (_, days, slug, stage, count, last, now, drop, zero) in enumerate(rows):
        y = 0.75 + i * row_h + row_h / 2
        ax.plot([0.2, W - 0.2], [y - row_h / 2, y - row_h / 2], color="#eaeef2", lw=0.8)
        ax.text(0.3, y - 0.1, slug, fontsize=10.5, color=INK, va="center")
        ax.text(0.3, y + 0.15, stage, fontsize=8.5, color=MUTED, va="center")
        ax.text(3.1, y - 0.1, fmt_day(last, today), fontsize=10, color=INK, va="center")
        ax.text(3.1, y + 0.15, f"всего повторов: {count}", fontsize=8.5, color=MUTED, va="center")
        lost = min(count, up.FULL) - now                       # сколько «растаяло» за простой
        for k in range(up.FULL):                               # шкала из 10 делений
            color = STATE_COLOR["закрыто"] if k < now else ("#c6e9d0" if k < now + lost else "#ebedf0")
            ax.add_patch(FancyBboxPatch((5.0 + k * 0.22, y - 0.12), 0.18, 0.24, boxstyle="round,pad=0,rounding_size=0.04",
                                        facecolor=color, edgecolor="none"))
        ax.text(7.3, y, f"{now}/{up.FULL}", fontsize=10.5, fontweight="bold", color=INK, va="center")
        if now == 0:
            ax.text(7.9, y, "уже на нуле — пора повторить", fontsize=9.5, color=BROKEN, va="center")
        else:
            soon = days <= 7                                   # меньше недели — подсвечиваем
            ax.text(7.9, y - 0.1, fmt_day(drop, today), fontsize=10, color=BROKEN if soon else INK, va="center")
            ax.text(7.9, y + 0.15, f"через {days} дн.", fontsize=8.5, color=BROKEN if soon else MUTED, va="center")
            ax.text(10.0, y - 0.1, fmt_day(zero, today), fontsize=10, color=INK, va="center")
            ax.text(10.0, y + 0.15, "если не повторять", fontsize=8.5, color=MUTED, va="center")
    legend_y = 0.05 + list_h + 0.32                            # подпись к шкале под карточкой
    for k, (color, text) in enumerate([(STATE_COLOR["закрыто"], "знаю сейчас"), ("#c6e9d0", "растаяло за простой"),
                                       ("#ebedf0", "до цели 10")]):
        ax.add_patch(FancyBboxPatch((0.3 + k * 2.4, legend_y - 0.1), 0.18, 0.2, boxstyle="round,pad=0,rounding_size=0.04",
                                    facecolor=color, edgecolor="none", clip_on=False))
        ax.text(0.55 + k * 2.4, legend_y, text, fontsize=8.5, color=MUTED, va="center")
    return fig


# ───────────────────────────── 7. упали тесты ─────────────────────────────

def failed_by_topic(d, problems):
    """{(тема, кто): [проверки]} — кто: «эталон» или «попытка attempt_NN»."""
    info = {str(Path(p).parent): (s, st, sub) for p, s, st, sub in zip(d.path, d.slug, d.stage, d.subsection)}
    out = {}
    for name, file in problems.get("failed_tests", []):
        slug, stage, sub = info.get(str(Path(file).parent), (Path(file).parent.name, "", ""))
        pid = name.split("[", 1)[1].rstrip("]") if "[" in name else "etalon"
        who = "эталон" if pid == "etalon" else f"попытка {pid}"
        out.setdefault((slug, stage, sub, who), []).append(name.split("[", 1)[0])
    return out


def chart_failed(d, problems):
    """Упавшие тесты по-человечески: какая тема, что сломано (эталон или попытка), какие проверки не прошли."""
    groups = failed_by_topic(d, problems)
    if not groups:
        fig, rect = frame("Упали тесты", 1.5)
        ax, W, H = inch_axes(fig, rect)
        ax.text(0.1, H / 2, "✓  Все тесты проходят", fontsize=13, fontweight="bold",
                color=STATE_COLOR["закрыто"], va="center")
        return fig
    lines = {key: textwrap.wrap(", ".join(tests), 78) for key, tests in groups.items()}
    heights = [0.62 + 0.24 * max(len(v) - 1, 0) for v in lines.values()]
    fig, rect = frame("Упали тесты", sum(heights) + 0.35 + 0.95,
                      note="Тема получает «!» на карте, если не прошла хоть одна проверка эталона или попытки.")
    ax, W, H = inch_axes(fig, rect)
    white_card(ax, 0, 0.05, W, sum(heights) + 0.2)
    y = 0.2
    for ((slug, stage, sub, who), wrapped), rh in zip(lines.items(), heights):
        ax.scatter(0.35, y + 0.2, s=260, color=BROKEN, zorder=3)
        ax.text(0.35, y + 0.2, "!", ha="center", va="center", fontsize=11, fontweight="bold", color="white", zorder=4)
        ax.text(0.7, y + 0.12, slug, fontsize=11, color=INK, va="center", fontweight="bold")
        ax.text(0.7, y + 0.36, f"{stage} · {sub}", fontsize=8.5, color=MUTED, va="center")
        ax.add_patch(FancyBboxPatch((3.0, y + 0.02), len(who) * 0.075 + 0.3, 0.3, boxstyle="round,pad=0,rounding_size=0.1",
                                    facecolor="#ffebe9", edgecolor="none"))   # плашка: что сломано
        ax.text(3.15, y + 0.17, who, fontsize=9, color=BROKEN, va="center")
        ax.text(5.2, y + 0.17, f"не прошли {len(groups[(slug, stage, sub, who)])}:", fontsize=9, color=MUTED, va="center")
        for k, line in enumerate(wrapped):
            ax.text(6.5, y + 0.17 + k * 0.24, line, fontsize=9, color=INK, va="center")
        y += rh
    return fig


# ───────────────────────────── 8. не указан статус ─────────────────────────────

def chart_no_status(d):
    """Темы без статуса: подсказка, что вписать, и сами темы колонками по этапам."""
    bad = d[d.state == "без статуса"]
    if bad.empty:
        fig, rect = frame("Не указан статус", 1.5)
        ax, W, H = inch_axes(fig, rect)
        ax.text(0.1, H / 2, "✓  У всех тем есть статус", fontsize=13, fontweight="bold",
                color=STATE_COLOR["закрыто"], va="center")
        return fig
    n_cols, row_h = 3, 0.3
    blocks = []                                                # (этап, [(тема, подраздел)]) — по 3 в строке
    for stage in up.ORDER:
        g = bad[bad.stage == stage]
        if len(g):
            blocks.append((stage, list(zip(g.slug, g.subsection))))
    body_h = sum(0.45 + row_h * -(-len(items) // n_cols) + 0.12 for _, items in blocks)
    fig, rect = frame(f"Не указан статус · {len(bad)}", body_h + 1.95)
    ax, W, H = inch_axes(fig, rect)
    ax.text(0.1, 0.2, "Впиши в шапку файла темы одну строку:", fontsize=9.5, color=INK, va="center")
    for k, (code, meaning) in enumerate([("# status: todo", "не начинал"), ("# status: in_progress", "в работе"),
                                         ("# status: ready", "закрыл")]):
        x = 3.45 + k * 2.85
        ax.add_patch(FancyBboxPatch((x, 0.06), len(code) * 0.072 + 0.24, 0.28, boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor="white", edgecolor=GRID, linewidth=0.8))
        ax.text(x + 0.12, 0.2, code, fontsize=9, family="monospace", color=INK, va="center")
        ax.text(x + len(code) * 0.072 + 0.36, 0.2, meaning, fontsize=8.5, color=MUTED, va="center")
    white_card(ax, 0, 0.5, W, body_h + 0.1)
    y, col_w = 0.62, (W - 0.6) / n_cols
    for stage, items in blocks:
        ax.text(0.3, y + 0.15, f"{stage.upper()}  ·  {len(items)}", fontsize=9, fontweight="bold", color=MUTED,
                va="center")
        y += 0.42
        for k, (slug, sub) in enumerate(items):
            r, c = divmod(k, n_cols)
            x, yy = 0.3 + c * col_w, y + r * row_h
            ax.scatter(x + 0.06, yy, s=30, color=STATE_COLOR["без статуса"])
            ax.text(x + 0.2, yy, slug, fontsize=10, color=INK, va="center")
            ax.text(x + 1.85, yy, sub, fontsize=8.5, color=FAINT, va="center")   # подраздел — ровной колонкой
        y += row_h * -(-len(items) // n_cols) + 0.12
    return fig


# ───────────────────────────── всё сразу ─────────────────────────────

def build_all(df, problems, root):
    """Все графики по порядку: [(имя, функция без аргументов)]. Рисуются по одному — не копятся в памяти."""
    d = prepare(df)
    maps = lambda i: lambda: chart_map(df)[i]                  # карта: часть 1 и часть 2
    return [("1_summary", lambda: chart_summary(d, df)), ("2_map_part1", maps(0)), ("3_map_part2", maps(1)),
            ("4_matrix", lambda: chart_matrix(d)), ("5_focus", lambda: chart_focus(d)),
            ("6_radar", lambda: chart_radar(df)), ("7_repeats", lambda: chart_repeats(d, problems, root)),
            ("8_failed", lambda: chart_failed(d, problems)), ("9_no_status", lambda: chart_no_status(d))]


def to_png(item, dpi=110):
    """Фигура → PNG-байты и сразу закрыть её (карта уже приходит байтами)."""
    import io
    if isinstance(item, bytes):
        return item
    buf = io.BytesIO()
    item.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(item)
    return buf.getvalue()


def show_all(df, problems, root):
    """Рисует все графики в ноутбуке по одному: создать → показать → закрыть."""
    from IPython.display import Image, display
    plt.close("all")                                           # убрать фигуры, оставшиеся от прошлых запусков
    for _, make in build_all(df, problems, root):
        display(Image(to_png(make())))


def save_all(df, problems, root, out_dir):
    """То же, что show_all, но в файлы PNG (для проверки без ноутбука)."""
    plt.close("all")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for name, make in build_all(df, problems, root):
        (out_dir / f"{name}.png").write_bytes(to_png(make()))
        names.append(name)
    return names


def save_dashboard(df, problems, root, out_path, width=1600, gap=24):
    """Все графики одной картинкой сверху вниз (assets/dashboard.png) — общая сводка после каждого обновления."""
    import io
    from PIL import Image as PILImage                          # Pillow (работа с картинками) — склеиваем PNG
    plt.close("all")
    parts = []
    for _, make in build_all(df, problems, root):
        img = PILImage.open(io.BytesIO(to_png(make(), dpi=140))).convert("RGB")
        parts.append(img.resize((width, round(img.height * width / img.width)), PILImage.LANCZOS))   # общая ширина
    sheet = PILImage.new("RGB", (width, sum(p.height for p in parts) + gap * (len(parts) + 1)), "white")
    y = gap
    for part in parts:
        sheet.paste(part, (0, y))
        y += part.height + gap
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet = sheet.quantize(colors=256, method=PILImage.Quantize.MEDIANCUT)   # 256 цветов: на глаз то же, файл в разы меньше
    sheet.save(out_path, optimize=True)
