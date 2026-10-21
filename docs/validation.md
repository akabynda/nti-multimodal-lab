# Проверки

Команды выполняются из корня проекта в [активированном окружении](../README.md#установка-с-нуля).

## Тесты

```powershell
python -m pytest -q
node --check src/nti_lab/static/app.js
```

Pytest проверяет BCRX, обработку сигналов и пропусков, временные шкалы, аудио, разделение по участникам и API. Тест с локальным BCRX пропускается при отсутствии исходника. Проверка JavaScript требует Node.js.

## Контейнер

```powershell
docker build -t nti-lab .
docker run --rm nti-lab python -m pytest -q
```

[GitHub Actions](../.github/workflows/ci.yml) запускает тесты на Python 3.12 и сборку контейнера при push и pull request.

## Аудит результатов

```powershell
python scripts/audit_audio.py --output outputs/audio --expected-recordings 9
python scripts/audit_project.py
```

Замените `9` на число записей своего набора.

| Проверка | Результат |
|---|---|
| Стенограммы, временные границы, акустика, диаризация | `outputs/audio/audit.json` |
| Длительности, временные шкалы, наличие модальностей | `outputs/audit/project.json` |

В аудите проекта проверьте поля `status` и `errors`. Статус `incomplete` означает недостающие данные или несоответствия.

## Интерфейс

После [запуска](../README.md#быстрый-запуск-в-подготовленной-папке-windows) проверьте выбор записи, покадровый шаг, переход по таймкоду истории, воспроизведение, паузу, наложение ROI и консоль браузера.
