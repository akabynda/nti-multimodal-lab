# Исходные данные

`NTI_DATA_ROOT` задаёт каталог данных; по умолчанию используется родительская папка проекта.

```text
dataset/
  Participant01.bcrx
  Session1.bcsx
  Результаты.xlsx
  1_30/
    00001_Participant01.MTS
  nti-lab/
```

`recording_id` — имя видео без расширения. `participant_id` — часть после первого подчёркивания; она связывает видео с `outputs/bio/<participant_id>`. Поддерживаются MTS и MP4. Для BCRX с другим именем задайте ID:

```powershell
python scripts/analyze_bio.py --input data/private/Participant09/recording.bcrx --subject-id Participant09
```

Состав набора: девять видео, девять BCRX, 54 истории и таблица профилей. Каналы и частоты описаны в [формате BioRadio](../docs/bioradio_format.md).

## Локальные файлы

| Путь в `data/private/` | Содержание |
|---|---|
| `source_documents.json` | Извлечённые тексты, идентификаторы, даты и URL документов |
| `source_files/` | Исходные PDF/DOCX |
| `catalog_spec.json` | Границы тем и соответствие участников номерам ИСУ |
| `stories.json` | Истории, источники, SHA-256 текста и диапазоны символов |
| `reference_truth.json` | Подтверждение истинности письменных эталонов |
| `reviewed_trials.csv` | Интервалы ответов и экспертные метки |
| `annotations/` | Разметка интерфейса |

Профили присоединяются по номеру ИСУ и столбцу «Таб.» книги XLSX. Сохраняются исходные категории, лист, строка и SHA-256 книги. Каталог `private/` исключён из Git.

Пример записи в `stories.json`:

```json
{
  "stories": [{
    "story_id": "Participant01_story_01",
    "participant_id": "Participant01",
    "title": "Тема",
    "text": "Исходный рассказ",
    "source_url": null,
    "declared_truth_label": "truth",
    "label_provenance": "подтверждение автора набора"
  }]
}
```
