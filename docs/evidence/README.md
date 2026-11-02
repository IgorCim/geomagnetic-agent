# Historical evidence: Kaggle `stress_3ffd1r21`

## Что найдено

Локально найден скачанный Kaggle notebook:

- файл: `C:\Users\User\Downloads\10112007 (1).ipynb`;
- Kaggle notebook `10112007`; owner в URL заменён на `<owner>`, чтобы не тиражировать логин. Точная ссылка есть в истории браузера и в самом локальном notebook;
- SHA-256 исходного notebook: `204775827ee43618c286c156b2ffd7df8ac2c7b0c9f1d2a4b2f04826a789c4d0`;
- размер: 95 426 байт.

В нём сохранены outputs двух stress-прогонов:

| Ячейка | RUN_DIR | execution_count | Время UTC | Итог |
|---|---|---:|---|---|
| 10 | `stress_3ffd1r21` | 7 | 2026-10-28 15:12:38–15:16:57 | 300/300, SuperMAG 2×21 600 строк, `skipped: none`, `failures: none` |
| 9 | `stress_ke1piibk` | 16 | 2026-10-28 05:23:47–05:28:07 | те же заявленные результаты |

Именно `stress_3ffd1r21` указан в [ACCEPTANCE.md](../../ACCEPTANCE.md).

### Хронология

| Время UTC | Событие |
|---|---|
| 2026-10-28 05:23–05:28 | output `stress_ke1piibk` (ячейка 9) |
| 2026-10-28 09:31 | коммит `ed46cf7` |
| 2026-10-28 15:12–15:16 | output `stress_3ffd1r21` (ячейка 10) |
| 2026-10-28 17:16 | коммит `16a5b0a` |
| 2026-10-28 17:37 | ячейка 8 assert на HEAD `16a5b0a` |

`stress_3ffd1r21` по времени находится между `ed46cf7` и `16a5b0a`. Это согласуется с baseline `ed46cf7` из ACCEPTANCE, но не доказывает его: сам stress-output HEAD не печатает.

## Что подтверждает найденный output

В ячейке 10 есть строки:

```text
[4/9 SuperMAG indices] ok=True elapsed=16.6s
Выполнено 2 из 2 подзадач. Ошибок: 0.
- SuperMAG indices: 21600 строк (2024-05-01..2024-05-15) -> 'idx:supermag:2024-05-01'.
- SuperMAG indices: 21600 строк (2024-05-16..2024-05-30) -> 'idx:supermag:2024-05-16'.
```

Итоговый блок:

```text
skipped: none
security probe: ok
security notes: none
expected service notes:
  - SuperMAG: exercised with SUPERMAG_LOGON
failures: none
STRESS TEST PASSED: 300 distinct canonical charts, 6 sources exercised, 3 matrix runs, 3 default ZIPs, 1 prompt-named custom ZIP
```

Это подтверждает, что в этом сохранённом Kaggle-output:

- матрицы A/B/C заявлены как 200/200, 50/50 и 50/50;
- два окна SuperMAG вернули по 21 600 строк и `ok=True`;
- `skipped: none` и `failures: none`;
- security probe прошёл.

## Чего этот output не доказывает

1. **Это не независимый raw stdout.** Это сохранённые outputs notebook. В нём нет ZIP/CSV, нет хешей артефактов и нет отдельного файла лога.
2. **Два прогона очень близки, но не идентичны.** У `stress_3ffd1r21` и `stress_ke1piibk` почти совпадает структура и длительность, но stage timings отличаются на 0.1–0.7 с: `126.1` vs `127.1`, `36.4` vs `36.7`, `13.2` vs `13.7`, `10.2` vs `10.3`, `2.1` vs `2.2`, `31.7` vs `31.4`, `19.4` vs `19.3`, `2.1` vs `2.2`. Также в них разные NOAA forecast (Kp 5.0 vs 4.0), горизонт ISWA (15:00 vs 05:00) и в `stress_3ffd1r21` есть `security probe: ok`, которого нет в `stress_ke1piibk`. Это похоже на два реальных прогона на одной нагрузке, но независимость по notebook не доказана; нельзя называть их двумя независимыми подтверждениями без raw logs и артефактов.
3. **Привязка к коммиту не доказана, но временная линия согласуется с `ed46cf7`.** `stress_3ffd1r21` выполнен в 15:12 UTC, между коммитами `ed46cf7` (09:31 UTC) и `16a5b0a` (17:16 UTC). Ячейка 8 с assert на HEAD `16a5b0a` выполнена позже, в 17:37 UTC. Сам stress-output HEAD не печатает, поэтому это разумная реконструкция, а не машинное доказательство.
4. **Строки и статусы не равны научной корректности.** Лог не показывает значения SME/SML/SMU, не проверяет единицы, пропуски и согласованность кривых.
5. **Это историческое свидетельство, не текущая доступность.** Сейчас старый SuperMAG endpoint отвечает HTTP 404; текущий статус нельзя подменять этим логом.

## Безопасность

В исходном notebook ячейка 2 содержит hardcoded `SUPERMAG_LOGON`. **Не публикуйте `10112007 (1).ipynb` как есть.** В извлечённых логах значение заменено на `[REDACTED]`; исходный notebook остаётся локально и не копируется в репозиторий.

Перед любой публикацией отдельно проверьте:

- нет ли логина в истории ячеек, traceback, URL и метаданных;
- не попали ли ZIP/CSV с персональными путями;
- не выдан ли historical output за текущий live-прогон.

## Файлы

- `stress_3ffd1r21_cell10_output_redacted.log` — output ячейки 10, redacted.
- `stress_ke1piibk_cell9_output_redacted.log` — output ячейки 9, redacted.
- `stress_kaggle_summary.json` — машиночитаемая сводка, хеши и ограничения.

## Как перепроверить

```powershell
@'
import json, hashlib
from pathlib import Path
p = Path(r'C:\Users\User\Downloads\10112007 (1).ipynb')
raw = p.read_bytes()
print(hashlib.sha256(raw).hexdigest())
nb = json.loads(raw.decode('utf-8'))
for i in (9, 10):
    c = nb['cells'][i]
    print(i, c.get('execution_count'), c.get('metadata', {}).get('execution'))
'@ | python -
```

Ожидаемый SHA-256 должен совпасть с указанным выше. Если файл изменился, хеш и выводы этого каталога больше не относятся к нему.

## Вывод

Найденный notebook — полезное **историческое** свидетельство того, что старый контур действительно доходил до SuperMAG и заявлял 300/300. Но это не чистый независимый лог и не основание для текущей приёмки 6/6. Для портфолио его можно показывать только с подписью:

> Historical Kaggle output, 28.10.2026. SuperMAG was reachable then; current endpoint returns 404. Saved notebook output, not a current live run.
