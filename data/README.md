# contributions.csv

The source for the activity images on the profile page (the heatmap, the streak card and the 30-day graph).

| column | meaning |
|---|---|
| `date` | calendar day, `YYYY-MM-DD` (Asia/Almaty) |
| `count` | contributions that day |
| `source` | `auto`: filled in by the robot · `manual`: your value, never touched |

This file lives on the **`stats`** branch, so the main page of the repo shows only the README.

**To fix a day**, open [`data/contributions.csv` on the `stats` branch](https://github.com/AlisherKshtykenov/AlisherKshtykenov/blob/stats/data/contributions.csv), press ✏️, set the number and write `manual`:

```
2026-11-24,13,manual
```

After you save, the images redraw within a couple of minutes.
Every 6 hours the robot adds the new day, recounts the last 7 days for `auto` rows (it only ever raises them), and redraws the images.

---

**По-русски.** Здесь лежат данные для графиков активности в профиле. Файл лежит в ветке `stats`. Чтобы поправить день, открой его там, нажми ✏️, измени число в строке и поставь `manual`, тогда робот эту строку больше не трогает. Новый день добавляется сам, картинки перерисовываются через пару минут после сохранения.
