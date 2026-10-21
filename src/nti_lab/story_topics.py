"""Consistent display titles; source text and source identifiers stay intact."""
from __future__ import annotations

import re


def story_topic_title(title: str) -> str:
    clean = re.sub(r"^\s*(?:\*+\s*)?(?:\d+\s*[.)]\s*)?", "", title or "")
    clean = clean.strip(' \t\r\n*«»“”\"\'.,?!…')
    key = clean.casefold().replace("ё", "е")
    for fragment, name in (
        ("професс", "Выбор профессии"),
        ("экзамен", "Экзамен"),
        ("достижен", "Достижения"),
        ("алкогол", "Алкоголь"),
        ("обман", "Обман"),
        ("конфликт", "Конфликт"),
        ("конфлит", "Конфликт"),
        ("проблема, с которой мне не удалось справиться", "Нерешённая проблема"),
    ):
        if fragment in key:
            return name
    if "день" in key and "лет" in key:
        return "Летний день"
    # This source heading belongs to a story explicitly describing a work conflict.
    if key == "напарник, который я больше всего запомнила":
        return "Конфликт"
    return clean


def display_story_titles(record: dict) -> dict:
    """Project titles for views/exports while retaining the exact original heading."""
    result = dict(record)
    for field in ("title", "story_title"):
        if field in record:
            original = record.get(f"original_{field}", record[field])
            result[field] = story_topic_title(original)
            result[f"original_{field}"] = original
    return result
