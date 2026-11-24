#!/usr/bin/env python3
"""Self-hosted contribution stats for the profile README.

Data lives in data/contributions.csv (date,count,source). Rows with source=manual
are never touched by the robot, so you can fix any day by hand.

    python scripts/stats.py update [--days 7]   # recount recent days from GitHub, add today
    python scripts/stats.py render [--out dist]  # draw SVGs from the CSV

Standard library only. Token: GH_STATS_TOKEN (a PAT, sees private repos) or GITHUB_TOKEN
(public repos only). Without a token, `update` only adds empty days.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "contributions.csv"
USER = os.environ.get("STATS_USER", "AlisherKshtykenov")
AUTHOR_NAMES = {"КАШТАН"}  # commit author names that count as you, besides your login
TZ = ZoneInfo(os.environ.get("STATS_TZ", "Asia/Almaty"))

THEMES = {
    "dark": dict(bg="#0d1117", border="#30363d", empty="#161b22", text="#8b949e", title="#f0e6dc",
                 accent="#d97742", levels=["#3b1f0e", "#7a3b16", "#b5612a", "#d97742"]),
    "light": dict(bg="#ffffff", border="#d0d7de", empty="#ebedf0", text="#57606a", title="#24292f",
                  accent="#b5612a", levels=["#f5d0b0", "#e8a06b", "#d97742", "#7a3b16"]),
}
FONT = "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"


# ---------- data ----------

def today() -> date:
    return datetime.now(TZ).date()


def load() -> dict[date, tuple[int, str]]:
    rows: dict[date, tuple[int, str]] = {}
    if CSV_PATH.exists():
        with CSV_PATH.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("date"):
                    rows[date.fromisoformat(r["date"].strip())] = (int(r.get("count") or 0),
                                                                    (r.get("source") or "auto").strip())
    return rows


def save(rows: dict[date, tuple[int, str]]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "count", "source"])
        for d in sorted(rows):
            w.writerow([d.isoformat(), rows[d][0], rows[d][1]])


def api(path: str, token: str) -> list | dict:
    req = urllib.request.Request(f"https://api.github.com{path}",
                                 headers={"Authorization": f"Bearer {token}",
                                          "Accept": "application/vnd.github+json",
                                          "User-Agent": "profile-stats"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def paged(path: str, token: str):
    page = 1
    while True:
        sep = "&" if "?" in path else "?"
        items = api(f"{path}{sep}per_page=100&page={page}", token)
        if not items:
            return
        yield from items
        if len(items) < 100:
            return
        page += 1


def count_from_github(since: date, token: str) -> dict[date, int]:
    """Commits authored by you (default branches) + repositories created, per local day."""
    counts: dict[date, int] = {}
    since_iso = datetime.combine(since, datetime.min.time(), TZ).isoformat()
    # a PAT sees your private repos via /user/repos; the Actions token only sees public ones
    try:
        repos = list(paged("/user/repos?affiliation=owner", token))
    except Exception:
        repos = list(paged(f"/users/{USER}/repos?type=owner", token))
    for repo in repos:
        created = datetime.fromisoformat(repo["created_at"].replace("Z", "+00:00")).astimezone(TZ).date()
        if created >= since:
            counts[created] = counts.get(created, 0) + 1
        q = urllib.parse.urlencode({"since": since_iso})
        try:
            commits = list(paged(f"/repos/{repo['full_name']}/commits?{q}", token))
        except Exception:
            continue  # empty repo or no access
        for c in commits:
            login = (c.get("author") or {}).get("login", "")
            name = c["commit"]["author"]["name"]
            if login == USER or name in AUTHOR_NAMES:
                d = datetime.fromisoformat(c["commit"]["author"]["date"].replace("Z", "+00:00")).astimezone(TZ).date()
                counts[d] = counts.get(d, 0) + 1
    return counts


def update(days: int) -> None:
    rows = load()
    end = today()
    since = end - timedelta(days=days - 1)
    token = os.environ.get("GH_STATS_TOKEN") or os.environ.get("GITHUB_TOKEN")
    fresh = count_from_github(since, token) if token else {}
    d = min([since, *rows]) if rows else since
    while d <= end:
        old = rows.get(d)
        if old and old[1] == "manual":
            pass  # your hand-made value wins
        elif d >= since and token:
            # never lower a day: the Actions token can't see private repos, the seed could
            rows[d] = (max(fresh.get(d, 0), old[0] if old else 0), "auto")
        elif not old:
            rows[d] = (0, "auto")
        d += timedelta(days=1)
    save(rows)
    print(f"updated {since}..{end}: {sum(fresh.values()) if token else 'no token, gaps filled'}")


# ---------- stats ----------

def streaks(counts: dict[date, int], end: date):
    """(current, cur_start, cur_end, longest, long_start, long_end). Today at 0 doesn't break it."""
    longest = (0, None, None)
    run, start = 0, None
    d = min(counts) if counts else end
    while d <= end:
        if counts.get(d, 0) > 0:
            start = d if run == 0 else start
            run += 1
            if run > longest[0]:
                longest = (run, start, d)
        else:
            run = 0
        d += timedelta(days=1)
    cur_end = end if counts.get(end, 0) > 0 else end - timedelta(days=1)
    cur, d = 0, cur_end
    while counts.get(d, 0) > 0:
        cur += 1
        d -= timedelta(days=1)
    return cur, d + timedelta(days=1), cur_end, *longest


def fmt(d: date | None) -> str:
    return f"{d:%b} {d.day}" if d else "—"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------- svg: heatmap ----------

def heatmap(counts: dict[date, int], end: date, t: dict) -> str:
    cell, gap = 11, 3
    step = cell + gap
    left, top = 36, 46
    last_sunday = end - timedelta(days=(end.weekday() + 1) % 7)  # week starts Sunday, like GitHub
    first = last_sunday - timedelta(weeks=52)
    year_total = sum(v for d, v in counts.items() if first <= d <= end)
    nonzero = sorted(v for d, v in counts.items() if first <= d <= end and v > 0)

    def level(v: int) -> int:
        if v <= 0 or not nonzero:
            return 0
        q = [nonzero[min(len(nonzero) - 1, int(len(nonzero) * p))] for p in (0.25, 0.5, 0.75)]
        return 1 + sum(v > x for x in q)

    w, h = left + 53 * step + 16, top + 7 * step + 40
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           "<style>.c{animation:f .6s ease backwards}@keyframes f{from{opacity:.35}}</style>",
           f'<rect x=".5" y=".5" width="{w-1}" height="{h-1}" rx="10" fill="{t["bg"]}" stroke="{t["border"]}"/>',
           f'<text x="{left}" y="24" fill="{t["title"]}" style="{FONT};font-size:15px;font-weight:600">'
           f'{year_total} contributions in the last year</text>']
    for i, name in ((1, "Mon"), (3, "Wed"), (5, "Fri")):
        out.append(f'<text x="8" y="{top + i * step + 9}" fill="{t["text"]}" style="{FONT};font-size:10px">{name}</text>')
    prev_month = None
    for wk in range(53):
        col_start = first + timedelta(weeks=wk)
        if col_start.month != prev_month and col_start.day <= 7:
            out.append(f'<text x="{left + wk * step}" y="{top - 8}" fill="{t["text"]}" '
                       f'style="{FONT};font-size:10px">{col_start:%b}</text>')
            prev_month = col_start.month
        for dow in range(7):
            d = col_start + timedelta(days=dow)
            if d > end:
                continue
            v = counts.get(d, 0)
            lv = level(v)
            color = t["empty"] if lv == 0 else t["levels"][lv - 1]
            out.append(f'<rect class="c" style="animation-delay:{wk * 18}ms" x="{left + wk * step}" '
                       f'y="{top + dow * step}" width="{cell}" height="{cell}" rx="2" fill="{color}">'
                       f'<title>{v} on {d.isoformat()}</title></rect>')
    lx = w - 16 - 5 * step - 62
    ly = top + 7 * step + 18
    out.append(f'<text x="{lx}" y="{ly + 9}" fill="{t["text"]}" style="{FONT};font-size:10px">Less</text>')
    for i, color in enumerate([t["empty"], *t["levels"]]):
        out.append(f'<rect x="{lx + 28 + i * step}" y="{ly}" width="{cell}" height="{cell}" rx="2" fill="{color}"/>')
    out.append(f'<text x="{lx + 32 + 5 * step}" y="{ly + 9}" fill="{t["text"]}" style="{FONT};font-size:10px">More</text>')
    out.append("</svg>")
    return "\n".join(out)


# ---------- svg: streak card ----------

def streak_card(counts: dict[date, int], end: date, t: dict) -> str:
    total = sum(counts.values())
    first_day = min((d for d, v in counts.items() if v > 0), default=end)
    cur, cs, ce, lng, ls, le = streaks(counts, end)
    w, h = 495, 195
    cx = [w / 6, w / 2, 5 * w / 6]

    def big(x, y, s, color, size=28):
        return (f'<text x="{x}" y="{y}" text-anchor="middle" fill="{color}" '
                f'style="{FONT};font-size:{size}px;font-weight:700">{s}</text>')

    def small(x, y, s, color, size=14, weight=400):
        return (f'<text x="{x}" y="{y}" text-anchor="middle" fill="{color}" '
                f'style="{FONT};font-size:{size}px;font-weight:{weight}">{esc(s)}</text>')

    cur_range = f"{fmt(cs)} – {fmt(ce)}" if cur > 1 else (fmt(ce) if cur == 1 else "—")
    lng_range = f"{fmt(ls)} – {fmt(le)}" if lng > 1 else fmt(ls)
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        "<style>.r{animation:p 2.4s ease-in-out infinite}@keyframes p{50%{stroke-opacity:.55}}</style>",
        f'<rect x=".5" y=".5" width="{w-1}" height="{h-1}" rx="10" fill="{t["bg"]}" stroke="{t["border"]}"/>',
        f'<line x1="{w/3}" y1="28" x2="{w/3}" y2="{h-28}" stroke="{t["border"]}"/>',
        f'<line x1="{2*w/3}" y1="28" x2="{2*w/3}" y2="{h-28}" stroke="{t["border"]}"/>',
        big(cx[0], 92, total, t["title"]),
        small(cx[0], 122, "Total Contributions", t["title"]),
        small(cx[0], 146, f"{fmt(first_day)}, {first_day.year} – Present", t["text"], 12),
        f'<circle class="r" cx="{cx[1]}" cy="78" r="40" fill="none" stroke="{t["accent"]}" stroke-width="5" '
        f'transform="rotate(-90 {cx[1]} 78)"/>',
        big(cx[1], 89, cur, t["title"]),
        small(cx[1], 146, "Current Streak", t["accent"], 14, 700),
        small(cx[1], 168, cur_range, t["text"], 12),
        big(cx[2], 92, lng, t["title"]),
        small(cx[2], 122, "Longest Streak", t["title"]),
        small(cx[2], 146, lng_range, t["text"], 12),
        "</svg>",
    ])


# ---------- svg: 30-day graph ----------

def graph(counts: dict[date, int], end: date, t: dict, days: int = 30) -> str:
    w, h = 880, 280
    left, right, top, bottom = 52, 20, 44, 44
    pw, ph = w - left - right, h - top - bottom
    ds = [end - timedelta(days=days - 1 - i) for i in range(days)]
    vs = [counts.get(d, 0) for d in ds]
    vmax = max(vs) or 1
    nice = 10 ** math.floor(math.log10(vmax)) if vmax >= 10 else 1
    ymax = max(5, math.ceil(vmax * 1.1 / nice) * nice)
    ymax = math.ceil(ymax / 5) * 5
    pts = [(left + i * pw / (days - 1), top + ph - v / ymax * ph) for i, v in enumerate(vs)]

    def smooth(p):  # Catmull-Rom → cubic Bézier
        d = f"M{p[0][0]:.1f},{p[0][1]:.1f}"
        for i in range(len(p) - 1):
            p0, p1, p2 = p[i - 1] if i else p[i], p[i], p[i + 1]
            p3 = p[i + 2] if i + 2 < len(p) else p2
            c1 = (p1[0] + (p2[0] - p0[0]) / 6, min(top + ph, p1[1] + (p2[1] - p0[1]) / 6))
            c2 = (p2[0] - (p3[0] - p1[0]) / 6, min(top + ph, p2[1] - (p3[1] - p1[1]) / 6))
            d += f" C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}"
        return d

    line = smooth(pts)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           "",
           f'<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{t["accent"]}" '
           f'stop-opacity=".45"/><stop offset="1" stop-color="{t["accent"]}" stop-opacity="0"/></linearGradient></defs>',
           f'<rect x=".5" y=".5" width="{w-1}" height="{h-1}" rx="10" fill="{t["bg"]}" stroke="{t["border"]}"/>',
           f'<text x="{w/2}" y="26" text-anchor="middle" fill="{t["title"]}" style="{FONT};font-size:15px;font-weight:600">'
           f'Contributions in the last {days} days · {sum(vs)}</text>']
    for k in range(6):
        y = top + ph - k * ph / 5
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}" stroke="{t["border"]}" '
                   f'stroke-dasharray="{"0" if k == 0 else "3 4"}"/>')
        out.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" fill="{t["text"]}" '
                   f'style="{FONT};font-size:11px">{round(k * ymax / 5)}</text>')
    for i, d in enumerate(ds):
        if i % 2 == 0 or i == days - 1:
            out.append(f'<text x="{pts[i][0]:.1f}" y="{top+ph+20}" text-anchor="middle" fill="{t["text"]}" '
                       f'style="{FONT};font-size:11px">{d.day}</text>')
    out.append(f'<path d="{line} L{pts[-1][0]:.1f},{top+ph} L{pts[0][0]:.1f},{top+ph} Z" fill="url(#g)"/>')
    out.append(f'<path class="l" d="{line}" fill="none" stroke="{t["accent"]}" stroke-width="2.5"/>')
    for (x, y), d, v in zip(pts, ds, vs):
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{t["bg"]}" stroke="{t["accent"]}" '
                   f'stroke-width="2"><title>{v} on {d.isoformat()}</title></circle>')
    out.append("</svg>")
    return "\n".join(out)


def render(out_dir: Path) -> None:
    counts = {d: v for d, (v, _) in load().items()}
    end = today()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in THEMES.items():
        sfx = "" if name == "light" else "-dark"
        (out_dir / f"heatmap{sfx}.svg").write_text(heatmap(counts, end, t), encoding="utf-8")
        (out_dir / f"streak{sfx}.svg").write_text(streak_card(counts, end, t), encoding="utf-8")
        (out_dir / f"graph{sfx}.svg").write_text(graph(counts, end, t), encoding="utf-8")
    print(f"rendered 6 SVGs to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("update")
    u.add_argument("--days", type=int, default=7, help="recount this many recent days (default 7)")
    r = sub.add_parser("render")
    r.add_argument("--out", type=Path, default=ROOT / "dist")
    a = p.parse_args()
    update(a.days) if a.cmd == "update" else render(a.out)
