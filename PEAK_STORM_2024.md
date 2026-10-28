# PEAK STORM 2024 — supplement к 300-графиковому gate

**Дата:** 2026-10-28
**Назначение:** показать пик бури 10–12 мая 2024, который не попадает в
300-графиковый набор.
**Статус:** 20/20 графиков, ZIP собран, данные сверены.

---

## 1. Зачем supplement

300-набор:

- A: 2024-05-14 .. 2024-06-02;
- B: 2024-05-01 .. 2024-05-10;
- C: 2024-05-14 .. 2024-05-18.

Пик: **Dst −406 nT в 02:00 UT 11 мая**, Kp 9.0. B содержит onset 10 мая
(Dst −339 nT), но не 11 мая; A/C начинаются 14 мая. Поэтому supplement — это
**отдельный набор**, а не изменение 300 HTML и не замена `stress_g3solehz`.

## 2. Состав

| Блок | Станции | Даты | Графиков |
|---|---|---|---:|
| H/D/I, отдельные панели | BRW, CMO, SHU, FRD, BOU | 10, 11, 12 мая | 15 |
| H vs OMNI Bz/V, отдельные панели | BRW, CMO, SHU, FRD, BOU | 11 мая | 5 |
| **Итого** | 5 | — | **20** |

Станции: BRW (70.1°N), CMO (64.9°N), SHU (55.3°N), FRD (38.2°N),
BOU (40.1°N). BRW/CMO — сильный авроральный отклик, SHU/FRD/BOU — умеренный.

## 3. Артефакты

Каталог (локальный, `stress_runs/` в git ignore):

```text
stress_runs/peak_storm_2024_supplement/
  charts/                         20 × ~4.97 MB HTML
  data/                           derived-ряды станций, OMNI, Kyoto
  index.csv                       20 строк, SHA-256 и размеры
  index.html                      человекочитаемый индекс
  manifest.json                   counts, sources, ограничения
  README.md                       пояснение
  PEAK_STORM_2024_SUPPLEMENT.zip  31 337 091 байт, 31 запись
```

- 20 уникальных канонических имён; source-тег последний:
  - `BRW_2024-05-10_H-D-I_INTERMAGNET.html`;
  - `BRW_2024-05-11_H_vs_OMNI_2024-05-01_Bz-V_INTERMAGNET-OMNI.html`.
- ZIP: 20 HTML, `index.csv`, `index.html`, `manifest.json`, `README.md`,
  `data/`; `zipfile.testzip()` — без ошибок.
- `index.csv` отсортирован по `date, station, file`; SHA-256 каждого файла
  совпадает.

## 4. Проверка

- Dst min: **−406.0 nT**, 2024-05-11 02:00 UT.
- Kp max: **9.0**, 2024-05-11 00:00 UT.
- OMNI: 4320 минут, Bz min −47.45 nT (11 мая 00:42), V max 1027.6 км/с
  (12 мая 01:16).
- H/D/I-панели: 3 отдельных y-оси, подписи `H (nT)`, `D (deg)`, `I (deg)`.
- C-панели: 3 отдельных панели, подписи `H (nT)`, `Bz (nT)`, `V (km/s)`.
- C-заголовок: `BRW 11.05.2024 H vs OMNI (sw:omni:2024-05-01) Bz/V` —
  descriptor, не дата данных.
- Тесты: `1000 passed, 3 skipped`; targeted plot/naming — `78 passed`.

## 5. Ограничения

1. **SuperMAG SME/SML** локально не строились: `SUPERMAG_LOGON` не задан.
   В Kaggle-прогоне `stress_3ffd1r21` подтверждены 2 окна × 21 600 строк, но
   не численные значения.
2. NOAA alerts/forecast — текущее окно 2026, не май 2024; ISWA map —
   `service_decommissioned`, forecast — текущий.
3. Dst — Kyoto WDC на момент запроса, может быть provisional.
4. INTERMAGNET — `best-avail`; для публикации нужны definitive-ряды, если
   станция их уже опубликовала.
5. Supplement не заменяет полный 300-gate и не проверяет все 300 кривых.

## 6. Воспроизведение

```powershell
cd C:\Users\User\Desktop\geomagnetic_agent
python peak_storm_2024.py
```

Скрипт идемпотентно пересобирает supplement, используя кэш INTERMAGNET и
live OMNI/Kyoto. Он не пишет в проекты `stress_3ffd1r21` и не трогает
`stress_g3solehz`.
