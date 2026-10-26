# Лабораторная работа НТИ: видео, речь и биосигналы

Приложение для просмотра видео и стенограмм, сопоставления ответов с письменными историями, анализа ЭКГ, дыхания, rPPG и движений. Сервер — FastAPI, интерфейс — браузерный клиент.

## Установка с нуля

Python 3.12. Обработка выполняется на CPU; для ASR на GPU нужны CUDA 12 и cuDNN 9. Рекомендуемая RAM для обработки — 16–32 ГБ.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\Activate.ps1
```

В Linux/macOS активация: `source .venv/bin/activate`. Проверенные версии Windows-зависимостей: [requirements-windows-lock.txt](requirements-windows-lock.txt). Команды ниже выполняются из корня проекта в активированном окружении.

## Быстрый запуск в подготовленной папке Windows

Запустите `start.cmd` или выполните:

```powershell
python -m uvicorn nti_lab.app:app --host 127.0.0.1 --port 8765
```

Откройте [интерфейс](http://127.0.0.1:8765). `NTI_DATA_ROOT` задаёт каталог исходных данных, `NTI_PROJECT_ROOT` — каталог проекта с результатами.

## Обработка

Разместите исходники по [схеме данных](data/README.md), затем выполните:

```powershell
python scripts/analyze_bio.py
python -m nti_lab.audio --input ../1_30 --output outputs/audio --skip-existing
python scripts/analyze_video.py --input ../1_30 --workers 4
python scripts/make_proxies.py --input ../1_30
python scripts/estimate_sync.py
python scripts/analyze_research.py
python scripts/analyze_contrasts.py
python scripts/audit_project.py
python scripts/build_report.py
```

Результаты сохраняются в `outputs/`, PDF — в `outputs/report/Отчет НТИ.pdf`. Параметры команд доступны через `--help`. [configs/pipeline.json](configs/pipeline.json) хранит описание выполненного расчёта; параметры запуска задаются в командах.

## Интерфейс

1. Выберите участника и письменную историю.
2. Перейдите к фрагменту по таймкоду; кнопки «−1 кадр» и «+1 кадр» переключают кадры.
3. Назначьте роли говорящих и настройте смещение временных шкал.
4. Добавьте интервалы фона, вопросов, ответов, тем и условий `truth`/`lie`; укажите источник метки и статус проверки.

Разметка сохраняется в `data/private/annotations/`.

## Docker

```powershell
docker compose up --build -d
docker compose logs --tail 50
docker compose down
```

Адрес сервиса: `127.0.0.1:8765`. Порт задаётся в `docker-compose.yml`. Исходники монтируются для чтения; результаты, разметка и модели — в тома.

## Документация

| Файл | Содержание |
|---|---|
| [Исходные данные](data/README.md) | Каталоги, идентификаторы, схема истории |
| [BioRadio](docs/bioradio_format.md) | BCRX, фильтры, признаки, контроль качества |
| [Аудио](docs/audio_method.md) | ASR, акустика, диаризация, исправление стенограмм |
| [Методы](docs/methodology.md) | Сопоставление текстов, видео, синхронизация, анализ условий |
| [Совместный анализ](docs/multimodal_analysis.md) | Профили, признаки ответов, корреляции и классификация |
| [Проверки](docs/validation.md) | Тесты, аудит файлов, проверка интерфейса |

Код находится в `src/nti_lab/`, команды — в `scripts/`, тесты — в `tests/`. Исходные записи, персональные данные, результаты и модели исключены из Git.
