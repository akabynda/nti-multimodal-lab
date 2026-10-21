# Аудиоаналитика

`nti_lab.audio` декодирует аудиодорожку в моно 16 кГц. Время отсчитывается от первого декодированного аудиосэмпла.

## Запуск

```powershell
python -m nti_lab.audio --input ../1_30 --output outputs/audio --skip-existing
```

`--skip-existing` использует сохранённый ASR и пересчитывает акустику и разметку. Для раздельного запуска в двух терминалах:

```powershell
python -m nti_lab.audio --input ../1_30 --skip-existing --asr-only
python -m nti_lab.audio --input ../1_30 --skip-existing --features-only --wait-for-asr
```

Проход large-v3:

```powershell
python -m nti_lab.audio --input ../1_30 --output outputs/audio-large-v3 --model models/faster-whisper-large-v3 --beam-size 10 --batch-size 4 --asr-only --skip-existing
```

Запуск на CPU:

```powershell
python -m nti_lab.audio --input ../1_30 --model turbo --device cpu --compute-type int8 --batch-size 4 --skip-existing
```

GPU: CUDA 12/cuBLAS и cuDNN 9. Установка cuDNN в локальный каталог Windows:

```powershell
python -m pip install --target vendor --no-deps nvidia-cudnn-cu12==9.1.1.17
```

Модуль добавляет `vendor/nvidia/cudnn/bin` и `$env:CUDA_PATH/bin` в окружение процесса. Акустика использует `praat-parselmouth==0.4.6`.

## Методы

| Этап | Метод и параметры |
|---|---|
| ASR | Faster Whisper/CTranslate2, русский язык; large-v3: `int8_float16`, beam 10, batch 4; turbo: beam 5, batch 4 |
| Разбиение реплик | Пунктуация, пауза ≥650 мс или длина 15 с; связь с ASR-чанком через `source_chunk_id` |
| Речь и паузы | Silero VAD; длительности по объединению интервалов |
| Акустические окна | Длина 5 с, шаг 1 с |
| F0 | Автокорреляция Praat, 60–500 Гц |
| F1–F3 | Burg-formants, верхняя частота 5500 Гц, кадры с обнаруженным F0 |
| Энергия и спектр | dBFS, спектральный центр, наклон 100–4000 Гц в dB/октаву, доля клиппинга |
| Темп речи | WPM и число гласных ASR-текста в секунду |
| Кандидаты вопрос/ответ | Вопросительный знак в сегменте и следующий сегмент ответа |

Акустические признаки рассчитываются по всей смеси голосов окна. Времена вне записи ограничиваются её длительностью; оригиналы сохраняются в `raw_model_start/end`, флаг — `timing_clipped_to_recording`.

## Диаризация

WeSpeaker ResNet34 ONNX извлекает векторы голоса, нормализует их и разделяет KMeans на два кластера: seed 42, 30 инициализаций. Используется центральный фрагмент реплики до 6 с; реплики короче 0.7 с получают `unknown`. Без модели применяется резервная кластеризация стандартизованных статистик MFCC.

```powershell
python -m pip install sherpa-onnx==1.13.8
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('csukuangfj/speaker-embedding-models','wespeaker_en_voxceleb_resnet34.onnx',local_dir='models/speaker')"
python -m nti_lab.speakers --input ../1_30 --wait-for-acoustics --workers 3
```

Кластеры обозначаются A/B. `speaker_role_candidate` назначается по частоте вопросов при разнице ≥0.15, наличии ≥10 реплик и ≥10% распределённых реплик в каждом кластере. `speaker_role` остаётся `unconfirmed` до разметки. `speaker_similarity_margin` хранит разницу косинусных сходств с центрами кластеров.

Ручная разметка: `data/private/annotations/speakers/<recording_id>.json`. Пример:

```json
{
  "intervals": [
    {"start": 0, "end": 8.1, "speaker_id": "A", "speaker_role": "interviewer"},
    {"start": 8.1, "end": 12.5, "speaker_id": "B", "speaker_role": "respondent"}
  ]
}
```

Метка применяется к реплике, середина которой попадает в интервал.

## Исправление стенограмм

| Скрипт | Действие |
|---|---|
| `review_audio.py` | Сравнивает версии по времени; сохраняет `transcript_comparison.json` |
| `redecode_audio_review.py` | Повторно распознаёт фрагменты: large-v3, beam 10 и 1, температура 0, без VAD и текстовой подсказки; сохраняет результаты и хеши в `outputs/audio-large-v3/rechecks/` |
| `assemble_audio_analysis.py` | Пересчитывает кластеры и пары вопрос/ответ; переносит акустику и VAD той же записи с совпадающей длительностью либо рассчитывает заново |
| `apply_editorial_review.py` | Применяет журнал ролей, исправлений и объединений к снимку ASR; проверяет SHA-256 и исходный текст |

Журнал исправлений содержит исходный и новый текст, причину и хеш повторного распознавания. Письменные эталоны в исправлении ASR не участвуют. Неясные места сохраняются с отметкой проверки. Подтверждённое прослушивание отсутствует.

Исходные сегменты, ID и времена сохраняются в `segments`; объединённые реплики — в `turns` со ссылками `segment_ids`. Устаревшие метки слов отмечаются в `word_alignment_status`. Интерфейс и TXT используют `turns`; пары вопросов и ответов строятся по отредактированным ролям.

```powershell
python scripts/apply_editorial_review.py --source outputs/audio-editorial-20261021/source --decisions outputs/audio-editorial-20261021/decisions --output outputs/audio
python scripts/audit_audio.py --output outputs/audio --expected-recordings 9
```

Для нового снимка ASR нужен соответствующий журнал решений.

## Результаты

`recording_id` — имя медиа без расширения; `participant_id` — часть после первого подчёркивания. Времена заданы в секундах; `time_s` — середина акустического окна.

| Файл | Содержание |
|---|---|
| `<recording_id>.segments.json` | Стенограмма, слова, реплики, спикеры, VAD, пары вопрос/ответ, модель, хеши и настройки |
| `<recording_id>.raw.json` | Исходный ответ ASR |
| `<recording_id>.txt` | Стенограмма с таймкодами |
| `<recording_id>.features.csv` | Акустические окна |
| `<recording_id>.summary.json` | Сводка обработки |
| `editorial_review.json` | Сводка редакторских изменений |

Основной каталог — `outputs/audio/`. При смене версии предыдущая сохраняется в `outputs/audio-archive/`; `revision.json` содержит хеши текущих стенограмм, `promotion.json` — связь версий. Промежуточные исправления сохраняются в `reviewed/`, их сводка — в `validation_summary.json`.
