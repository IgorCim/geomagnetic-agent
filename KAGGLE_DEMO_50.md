# KAGGLE DEMO 50 — запуск live 14B

**Файл ноутбука:** `kaggle_demo_50.ipynb`  
**Цель:** живой Qwen 2.5 14B на Kaggle GPU → 50 графиков за 10 станций ×
14–18 мая 2024, ZIP, partial/empty, multi-turn.  
**Код:** GitHub `IgorCim/geomagnetic-agent`, commit `e7db373` или новее.

> Локально в этом окружении Kaggle-прогон выполнить нельзя: нет Kaggle CLI,
> API-ключей и GPU. Ноутбук готов к загрузке в Kaggle.

---

## 1. Настройка Kaggle

1. Откройте Kaggle → **Create → Notebook**.
2. **File → Import Notebook** → выберите `kaggle_demo_50.ipynb`.
3. Справа **Settings**:
   - **Accelerator:** `GPU T4 x2` (подойдёт и `P100`, 16 GB);
   - **Internet:** `On`;
   - **Persistence:** `Files`.
4. При необходимости **Add-ons → Secrets**:
   - `SUPERMAG_LOGON` — только если хотите показывать SuperMAG; для 50-chart
     demo не требуется.

## 2. Порядок ячеек

| Ячейка | Что делает | Время |
|---|---|---:|
| 1 | Клонирует GitHub, проверяет commit | ~10 сек |
| 2 | Ставит CUDA-wheel `llama-cpp-python` | 1–2 мин |
| 3 | Проверяет GPU и CUDA offload | ~10 сек |
| 4 | Скачивает и загружает Qwen 2.5 14B Q4_K_M | 5–12 мин |
| 5 | Проверяет prompt budget | ~1 сек |
| 6 | **Live matrix 10×5 → 50 HTML** | 10–20 мин |
| 7 | Проверяет `charts.json`/HTML, делает default + custom ZIP | 1–2 мин |
| 7a | *Optional:* deterministic peak supplement 10–12 мая | 1–2 мин |
| 8 | Показывает 3 HTML | — |
| 9 | ABG 13 May → ожидаемый `no_data` | ~1 мин |
| 10a | Security probe: `..` и `../../escape` | ~1 сек |
| 10 | Live multi-turn: H → D на одном `FrameStore` | 2–5 мин |
| 11 | *Optional:* Gradio UI с публичной ссылкой | — |

**Итого:** ~20–35 минут, если модель скачивается с нуля.

## 3. Ожидаемый результат

```text
projects/demo_storm_may2024/
  50 HTML
  charts.json          50 записей
  manifest.json
  <station>/<date>/raw.csv, derived.csv, manifest.json

exports/demo_storm_may2024.zip
  canonical tree, 50 HTML, ~76 MB

exports/storm_report_2024_demo.zip
  папки 1/2/3, 50 HTML, ~76 MB
```

Проверки, которые должны пройти в ячейке 7:

- `charts.json rows = 50`;
- `HTML in project = 50`;
- `created_at stable = True`;
- `tree stable = True`;
- `charts.json same = True`;
- оба ZIP: 50 chart HTML, `zip test = None`.

Custom ZIP:

- `1` — IRT, API, BSL, ABG — 20;
- `2` — BRW, SHU, FRD — 15;
- `3` — CMO, BOU, TUC — 15;
- `index.csv` — 50 строк, `archive_path` с `1/2/3`, `file` канонический.

## 4. Что показывать

1. **3 HTML** — H/D/I на отдельных панелях, подписи `nT`/`deg`.
2. **`charts.json`** — 50 записей, source/station/date/components.
3. **Custom ZIP** — `1/2/3`, но канонические имена не переименованы.
4. **ABG 13 May** — `no_data`, 7 попыток, без выдуманных данных.
5. **Multi-turn** — H, затем D из того же handle.
6. **Security** — `create_project("..")` и `archive_name="../../escape"`
   отклонены.
7. **Optional Gradio** — публичная share-ссылка.

## 5. Если что-то пошло не так

### GitHub clone / pull

Проверьте **Internet: On**. Если репозиторий недоступен, скачайте ZIP
`geomagnetic-agent` и загрузите его как Kaggle Dataset, затем скопируйте в
`/kaggle/working/geomagnetic-agent`. Ноутбук ожидает именно этот путь.

### CUDA-wheel

Если `llama_cpp` не видит CUDA:

- перезапустите ячейку 2;
- проверьте, что wheel `cu121` подходит образу Kaggle;
- не заменяйте его CPU-сборкой для live demo — будет очень медленно.

### Модель не влезает в VRAM

`n_ctx=16384` — целевой. В `agent_core.load_brain` есть fallback на `12288`;
если и это не влезает, используйте GPU с большей памятью. Для T4 16 GB
14B Q4_K_M + KV обычно помещается.

### Matrix дала меньше 50 графиков

Ноутбук явно печатает `⚠️` и **не** выдаёт это за успех. Детерминированный
fallback:

```python
!python demo_make_50.py
```

Но в отчёте нужно указать: «live 14B matrix не завершила 50, fallback
детерминированный». Не смешивайте эти два результата.

### Gradio

Ячейка 11 блокирует ноутбук; остановите её после показа. Если `share=True`
не работает, используйте вывод HTML напрямую из ячеек 8.

## 6. После демо

- Скачайте оба ZIP из `exports/`.
- Сохраните лог ноутбука (File → Download → `.ipynb`/`.log`).
- Не коммитьте 50 HTML и ZIP в git: они большие, `*.html` уже в `.gitignore`.
- Полный 300-gate на новом HEAD — отдельный прогон перед release, не перед
  demo.

## 7. Ограничения

- Это 50-chart demo, не 300-chart gate.
- 50 HTML содержат только INTERMAGNET H/D/I.
- Пик 11 мая и C-графики Bz/V — в `peak_storm_2024_supplement` (ячейка 7a).
- Dst/Kp — Kyoto WDC; Dst может быть provisional.
- NOAA/ISWA current-window не являются историческим маем 2024.
- SuperMAG SME/SML — только при `SUPERMAG_LOGON`.
