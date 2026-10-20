"""Build the local Russian laboratory report from calculated outputs.

Uses reportlab and an available Cyrillic TrueType font; private results are never
embedded into the repository. Run after processing and validation.
"""
from pathlib import Path
import argparse
import json
import csv
import math
from datetime import date
from collections import Counter
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, PageBreak, Image, KeepTogether)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output",type=Path,default=OUT/"report"/"Отчет НТИ.pdf")
args=parser.parse_args()
REPORT_DATE=date.today().strftime("%d.%m.%Y")

def read(p,default=None):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else default

def csvrows(p):
    if not p.exists():
        return []
    with p.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

font_dir=Path("C:/Windows/Fonts")
if font_dir.exists():
    regular,bold=font_dir/"arial.ttf",font_dir/"arialbd.ttf"
else:
    regular=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    bold=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
pdfmetrics.registerFont(TTFont("Body",str(regular)))
pdfmetrics.registerFont(TTFont("BodyBold",str(bold)))
pdfmetrics.registerFontFamily("Body",normal="Body",bold="BodyBold",italic="Body",boldItalic="BodyBold")
styles=getSampleStyleSheet()
for name in styles.byName:
    styles[name].fontName="Body"
styles.add(ParagraphStyle("ReportTitle",fontName="BodyBold",fontSize=24,leading=30,spaceAfter=18))
styles.add(ParagraphStyle("Section",fontName="BodyBold",fontSize=15,leading=20,spaceBefore=14,spaceAfter=10,keepWithNext=True))
styles.add(ParagraphStyle("Sub",fontName="BodyBold",fontSize=11,leading=15,spaceBefore=10,spaceAfter=5,keepWithNext=True))
styles.add(ParagraphStyle("TextRu",fontName="Body",fontSize=10,leading=14,spaceAfter=8))
styles.add(ParagraphStyle("SmallRu",fontName="Body",fontSize=8.4,leading=11,spaceAfter=6))
styles.add(ParagraphStyle("CellRu",fontName="Body",fontSize=8.2,leading=11,spaceAfter=0))
story=[]
def p(text,style="TextRu"):
    return Paragraph(escape(str(text)).replace("\n","<br/>"),styles[style])
def add(text,style="TextRu"):
    story.append(p(text,style))
def heading(text):
    add(text,"Section")
def table(headers,rows,widths):
    cells=[[p(x,"CellRu") for x in headers]]+[[p(x,"CellRu") for x in row] for row in rows]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e5ece7")),
        ("FONTNAME",(0,0),(-1,0),"BodyBold"),
        ("GRID",(0,0),(-1,-1),.4,colors.HexColor("#d5ddd7")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f6f8f5")]),
    ]))
    story.append(t);story.append(Spacer(1,10))
def timecode(t):
    if t is None:return "нет сопоставимого интервала"
    return f"{int(t//60):02d}:{t%60:05.2f}"
def trim(t,n=240):
    t=str(t or "")
    return t if len(t)<=n else t[:n].rsplit(" ",1)[0]+"…"
def num(v,d=1):
    try:return f"{float(v):.{d}f}" if math.isfinite(float(v)) else "н/д"
    except (TypeError,ValueError):return "н/д"
def interval(value):
    return "–".join(num(x,2) for x in value) if isinstance(value,list) and len(value)==2 else "н/д"
def percent(value):
    return num(float(value)*100)+" %" if value is not None else "н/д"

bio=read(OUT/"bio/summary.json",{}).get("recordings",[])
research=read(OUT/"research/summary.json",{})
review=read(OUT/"research/content_comparison_report.json",{})
notes=read(OUT/"research/content_comparison.json",{}).get("comparisons",[])
sources=read(ROOT/"data/private/stories.json",{}).get("stories",[])
audios=[read(p) for p in sorted((OUT/"audio").glob("*.segments.json"))]
videos=[read(p) for p in sorted((OUT/"video").glob("*.json"))]
alignments=[read(p) for p in sorted((OUT/"sync").glob("*.json")) if p.stem!="summary"]
by_bio={x["subject_id"]:x for x in bio}
ppg_records=[b for b in bio if b.get("ppg_analysis")]
by_video={x["participant_id"]:x for x in videos}
order=[a["participant_id"] for a in audios]
# Personal display names are supplied locally, never embedded in public code.
private_names=read(ROOT/"data/private/participant_names.json",{})
names={pid:private_names.get(pid,f"Участник P{i:02}") for i,pid in enumerate(order,1)}
total_windows=sum(b.get("window_count",0) for b in bio)
by_alignment={a.get("participant_id",a.get("subject_id")):a for a in alignments}
gsr_review_count=sum(b.get("gsr_quality",{}).get("status")=="flat_or_out_of_range_review_required" for b in bio)
relation={"consistent":"Согласуется с эталоном","different":"Другой пример или деталь","omitted":"Не рассказано","uncertain":"Неоднозначный ответ","contradiction_candidate":"Расхождение с эталоном"}
counts=Counter(n["relation"] for n in notes)
asr_validation=read(OUT/"audio/validation_summary.json",{})

add("Лабораторная работа НТИ","ReportTitle")
add("Анализ видео, речи и биосигналов","Section")
add(REPORT_DATE)
heading("Задача и результат")
add("В эксперименте участники рассказывали истории и намеренно изменяли часть ответов. Письменные истории служат правдивым эталоном. Устные ответы сопоставлены с эталоном по событиям, действиям, деталям и отрицаниям.")
add(f"Обработано {len(audios)} записей общей продолжительностью {sum(a.get('duration_s',0) for a in audios)/60:.2f} мин. Сверено {review.get('stories_reviewed',len(sources))} историй. Подготовлены стенограммы с временными метками, показатели речи, ЭКГ и дыхания, видеопульс, кинематика и локальный интерфейс для просмотра результатов.")
table(["Результат сверки","Число сопоставлений"],[[relation[k],str(counts[k])] for k in relation],[365,130])
add(f"Всего {len(notes)} сопоставлений. Одна история может содержать несколько проверяемых утверждений. Для каждого сохранены цитата эталона, устная реплика, вопрос и время на видео.")
heading("Материалы")
table(["Материал","Объём"],[
 ["Видео",f"{len(videos)} записей; один ракурс"],
 ["Письменные истории",f"{len(sources)} историй; {research.get('source_documents',0)} исходных документов"],
 ["Речь",f"{sum(len(a.get('segments',[])) for a in audios)} сегментов расшифровки"],
 ["БиоРадио",f"{len(bio)} записей; ЭКГ, RIP, GSR; контактный ФПГ в {len(ppg_records)} записи"],
 ["Вариабельность ритма",f"{total_windows} окон по 120 с с шагом 5 с"],
],[155,340])
story.append(PageBreak())
heading("Записи участников")
rows=[]
for i,a in enumerate(audios,1):
    b=by_bio.get(a["participant_id"],{})
    duration=by_alignment.get(a["participant_id"],{}).get("media_duration_s",a.get("duration_s"))
    difference=b.get("analysis_duration_s",0)-duration if duration is not None and b.get("analysis_duration_s") is not None else None
    rows.append([f"P{i:02}",names[a["participant_id"]],num(duration),num(b.get("analysis_duration_s")),num(difference)])
table(["ID","Участник","Видео, с","БОС, с","Разница, с"],rows,[35,190,90,90,90])
add("Видео, тексты и записи прибора объединены по именам файлов и данным документов. Обозначения P01–P"+f"{len(order):02}"+" на графиках соответствуют порядку этой таблицы.")
heading("Синхронизация")
add("БиоРадио запускали перед видеозаписью и останавливали после неё. Поэтому длительность БОС больше длительности видео. Разница задаёт допустимый диапазон опережения прибора: от 0 до LБОС − Lвидео.")
add("За начало видео принят первый декодированный кадр. Начальный и конечный фон с закрытыми глазами найдены по речевым инструкциям и геометрии глаз. В этих фрагментах сопоставлены rPPG и опорный пульсовой ряд по R-зубцам ЭКГ. Проверены альтернативные пики корреляции.")
add("Преобразование времени: tвидео = offset + scale × tБОС. В расчётах scale = 1. В таблице приведены автоматические оценки смещения; статус отражает однозначность найденного пика.")
statuses={"supported_unverified":"Выраженный пик","ambiguous_unverified":"Несколько вариантов","insufficient_signal":"Мало сигнала"}
table(["Участник","Статус","Диапазон опережения, с","Offset, с"],[
 [names[pid],statuses.get(by_alignment.get(pid,{}).get("status"),"Не рассчитано"),
 interval(by_alignment.get(pid,{}).get("lead_bounds_s")),
 num(by_alignment.get(pid,{}).get("bio_to_video_offset_s",(by_alignment.get(pid,{}).get("candidate") or {}).get("bio_to_video_offset_s")),2)]
 for pid in order],[135,140,110,110])
story.append(PageBreak())
heading("ЭКГ, дыхание и качество сигналов")
add("Файлы BCRX декодированы в численные ряды. Частоты дискретизации, единицы и коэффициенты прочитаны из заголовков. После фильтрации ЭКГ найдены R-зубцы, проверены RR-интервалы и рассчитаны ЧСС, SDNN, RMSSD, pNN50, VLF, LF, HF и LF/HF [1].")
add("Окно ВСР — 120 с, шаг — 5 с. Окна с разрывами или нестабильными RR исключены из итоговых оценок. ULF для такой длительности не рассчитывается, VLF используется как исследовательский показатель. Дыхательный пояс RIP обработан отдельно; EDR рассчитан по колебаниям амплитуды ЭКГ.")
table(["Участник","Окон","Прошли контроль","Медиана ЧСС"],[
 [f"P{i+1:02}",str(by_bio.get(pid,{}).get("window_count","—")),str(by_bio.get(pid,{}).get("usable_ecg_windows","—")),num(by_bio.get(pid,{}).get("median_hr_bpm"))]
 for i,pid in enumerate(order)],[80,110,145,160])
fig=OUT/"bio/aggregate_summary.png"
if fig.exists():story.append(Image(str(fig),width=495,height=207))
add("Рисунок 1. Качество ЭКГ и описательные показатели по участникам.","SmallRu")
add(f"В {gsr_review_count} из {len(bio)} записей GSR обнаружены плоские участки или значения вне рабочего диапазона. Эти данные сохранены с отметкой качества.")
if ppg_records:
    story.append(PageBreak())
    heading("Контактный ФПГ и ЭКГ")
    add("Контактный ФПГ записан вместе с ЭКГ одним прибором. Сравнение использует общую шкалу отсчётов БиоРадио. Пульсовые интервалы рассчитаны после фильтрации и контроля качества.")
    table(["Участник","Окон ФПГ / прошли QC","Сравнено окон","MAE ЧСС, уд/мин"],[
        [names.get(b["subject_id"],b["subject_id"]),
         f"{b['ppg_analysis'].get('window_count','—')} / {b['ppg_analysis'].get('usable_ppg_windows','—')}",
         str(b.get("ecg_ppg_comparison",{}).get("paired_windows","—")),
         num(b.get("ecg_ppg_comparison",{}).get("hr_mae_bpm"),2)] for b in ppg_records],[135,130,110,120])
    for b in ppg_records:
        c=b.get("ecg_ppg_comparison",{})
        add(f"{names.get(b['subject_id'],b['subject_id'])}: средняя разность ЧСС ФПГ − ЭКГ составляет {num(c.get('hr_bias_ppg_minus_ecg_bpm'),2)} уд/мин. Средняя абсолютная разность RMSSD — {num(c.get('rmssd_mae_ms'),2)} мс, SDNN — {num(c.get('sdnn_mae_ms'),2)} мс. В сравнение включены окна, прошедшие контроль обоих каналов.")
        fig=OUT/"bio"/b["subject_id"]/"ecg_ppg_comparison.png"
        if fig.exists():
            im=Image(str(fig));im.drawHeight=495*im.imageHeight/im.imageWidth;im.drawWidth=495;story.append(im)
            add("Рисунок 2. Контактный ФПГ, R-зубцы ЭКГ и сравнение частоты пульса на общей шкале прибора.","SmallRu")
else:
    add("Контактный ФПГ в исходных записях отсутствует; пульс по видео сопоставлен с ЭКГ.")
story.append(PageBreak())
heading("Видеоаналитика")
add("По точкам позы выделена область головы, затем Face Landmarker определил лицевые ориентиры [5]. Для rPPG использованы лоб и щёки. Рот и глаза отслежены отдельно. Размер кадра, частота и соотношение сторон сохранены для каждой записи.")
add("Видеопульс рассчитан методом POS [2], с фильтрацией 0,7–3,5 Гц. Частота оценена в окнах по 30 с с шагом 5 с. Спектральная концентрация использована для отбора окон. По положению головы, плеч, кистей, таза и видимых стоп рассчитаны двумерные координаты, скорости и углы; по колебаниям плеч — частота дыхательного движения.")
rows=[]
for i,pid in enumerate(order):
    v=by_video.get(pid,{})
    c=by_alignment.get(pid,{}).get("crossmodal_summary",{})
    rows.append([f"P{i+1:02}",percent(v.get("face_coverage")),
       f"{len(v.get('windows',[]))} / {sum(w.get('rppg_hr_bpm') is not None for w in v.get('windows',[]))} / {c.get('hr_paired_windows',0)}",
       num(c.get("hr_mae_bpm"),2),f"{num(c.get('resp_mae_bpm'),2)} (n={c.get('resp_paired_windows',0)})"])
table(["Участник","Кадры с ROI","Окна: всего / ЧСС / пары ЭКГ","MAE ЧСС, уд/мин","MAE дыхания, /мин"],rows,[60,85,140,95,115])
add("MAE — средняя абсолютная разность частот: видеопульс сопоставлен с ЭКГ, движение плеч — с RIP. Использованы автоматические смещения из таблицы синхронизации и окна с допустимым качеством; n — число пар дыхательных оценок.","SmallRu")
fig=OUT/"audit/video_quality.png"
if fig.exists():story.append(Image(str(fig),width=495,height=264))
add(f"Рисунок {3 if ppg_records else 2}. Доля кадров с лицевыми областями и доля окон с рассчитанной ЧСС rPPG.","SmallRu")
story.append(PageBreak())
heading("Распознавание речи и проверка текста")
models_used=", ".join(sorted({str(a.get("model","Whisper")) for a in audios}))
add(f"Для полной расшифровки использована модель {models_used} через faster-whisper [6]. Сохранены временные метки слов и сегментов. Речь повторно сопоставлена с предыдущей расшифровкой; отдельно проверены отрицания, короткие ответы, вопросы и фразы с ошибками распознавания.")
if asr_validation:
    table(["Проверка","Результат"],[[label,str(asr_validation[key])] for key,label in [
        ("recordings","Повторно распознано записей"),("reviewed_segments","Прочитано исходных сегментов"),
        ("text_correction_count","Исправлено текстов реплик"),("timing_only_count","Уточнено временных границ"),
        ("recovered_segments","Восстановлено пропущенных реплик"),("unresolved_count","Фрагменты с неуверенным распознаванием")]
        if key in asr_validation],[365,130])
    repeat=asr_validation.get("repeat_review",{})
    if repeat:
        add(f"Повторно распознано {repeat.get('clips',0)} коротких фрагментов, каждый двумя проходами. Исправления и исходные варианты сохранены в журнале.","SmallRu")
    examples=asr_validation.get("correction_examples",asr_validation.get("examples",[]))
    selected=[]
    for keyword in ["аспирантуру","академразницу","Как вы себя чувствуете?"]:
        example=next((e for e in examples if keyword.lower() in e.get("after","").lower()),None)
        if example and example not in selected:selected.append(example)
    for example in selected or examples[:3]:
        pid=example.get("participant_id") or example.get("recording_id","").split("_",1)[-1]
        add(f"{names.get(pid,pid)} · {timecode(example.get('start_s'))}: «{trim(example.get('before',''),100)}» → «{trim(example.get('after',''),150)}».","SmallRu")
add("По аудиодорожкам извлечены основной тон и его изменчивость, энергия, паузы, доля звучащей речи, спектральный центр, наклон и форманты. Темп рассчитан по числу слов и длительности реплик. Голоса разделены по речевым представлениям WeSpeaker.")
heading("Сопоставление с эталоном")
add("Для каждой истории найден соответствующий участок интервью. Проверены событие, действующее лицо, время, действие и отрицание. Короткие ответы рассмотрены вместе с предшествующим вопросом. Результаты приведены по участникам на следующих страницах.")
table(["Категория","Критерий"],[[relation[k],v] for k,v in [
 ("consistent","Устный ответ воспроизводит событие или деталь эталона."),
 ("contradiction_candidate","Устное утверждение расходится с письменным событием или деталью."),
 ("different","Приведён другой пример или дополнительная деталь."),
 ("omitted","Письменный эпизод не пересказан."),
 ("uncertain","Смысл ответа или отдельные слова остаются неоднозначными.")]], [175,320])
add("Полная таблица содержит исходные цитаты, реплики, контекст и временные границы. В отчёте показаны ключевые сопоставления; остальные доступны в приложении и CSV.")

for pid in order:
    story.append(PageBreak())
    heading(names.get(pid,pid))
    summ=next((s for s in review.get("participant_summaries",[]) if s["participant_id"]==pid),{})
    add(summ.get("content_summary") or summ.get("summary","Сопоставление устных ответов с письменными историями."))
    rows=[]
    for src in [s for s in sources if s["participant_id"]==pid]:
        related=[n for n in notes if n["story_id"]==src["story_id"]]
        cats=list(dict.fromkeys(relation.get(x["relation"],x["relation"]) for x in related))
        rows.append([src["title"],"; ".join(cats) or "Не найден соответствующий эпизод"])
    if rows:table(["Письменная история","Результат сверки"],rows,[240,255])
    rank={"contradiction_candidate":0,"uncertain":1,"different":2,"consistent":3,"omitted":4}
    chosen=sorted([n for n in notes if n["participant_id"]==pid and n.get("spoken_excerpt")],key=lambda n:(rank.get(n["relation"],9),-float(n.get("confidence",0))))[:3]
    for n in chosen:
        add(f"{relation.get(n['relation'],n['relation'])} · {timecode(n.get('start_s'))}–{timecode(n.get('end_s'))}","Sub")
        add(f"Эталон: «{trim(n.get('source_excerpt'),230)}»","SmallRu")
        if n.get("question_summary"):
            add(f"Смысл вопроса: {trim(n['question_summary'],230)}","SmallRu")
        elif n.get("question_context"):
            add(f"Вопрос: «{trim(n['question_context'],230)}»","SmallRu")
        add(f"Ответ: «{trim(n.get('spoken_excerpt'),230)}»","SmallRu")
        for extra in n.get("additional_spoken_evidence",[]):
            question=(f" {extra['question_context']}" if extra.get("question_context") else "")
            add(f"Следующая реплика · {timecode(extra.get('start_s'))}:{question} «{trim(extra.get('spoken_excerpt'),160)}»","SmallRu")
        if n.get("protocol_condition"):
            add(f"Условие блока: {n['protocol_condition']}.","SmallRu")
        add(trim(n.get("interpretation"),370),"SmallRu")

story.append(PageBreak())
heading("Статистический анализ")
add("Реализованы расчёт индивидуальных контрастов между правдивыми и ложными ответами, перестановочный тест, поправка Холма и bootstrap по участникам. Модель классификации — логистическая регрессия. Проверка выполняется с исключением одного участника; обработка признаков обучается внутри тренировочной выборки [3, 4].")
add("Проверяемые гипотезы: различаются ли контрасты признаков между типами темперамента; улучшает ли добавление темперамента классификацию ответов нового участника.")
add("В исходной таблице профилей представлена другая типология, а классический темперамент не измерен. Поэтому сравнение по четырём типам темперамента и точность такой модели не рассчитаны. Содержательные результаты текущего эксперимента представлены сопоставлениями ответов с эталонами.")
heading("Интерфейс и запуск")
add("Локальное приложение объединяет пять зон: временную шкалу; стенограмму и истории; биосигналы; видео с областями лица и позой; спектр ВСР и численные признаки. Переход по цитате открывает нужный фрагмент с вопросом. Разметка и смещение сохраняются в JSON.")
interface_path=OUT/"report/interface.png"
if interface_path.exists():
    im=Image(str(interface_path));scale=min(495/im.imageWidth,340/im.imageHeight)
    im.drawWidth=im.imageWidth*scale;im.drawHeight=im.imageHeight*scale;story.append(im)
    add(f"Рисунок {4 if ppg_records else 3}. Интерфейс просмотра и сопоставления записей.","SmallRu")
add("Запуск: start.cmd. Подготовлены инструкции для Windows/Linux/macOS, Dockerfile, docker-compose.yml, зависимости, тесты и GitHub Actions. Код опубликован отдельно от персональных исходных данных.")
integrity=read(OUT/"audit/project.json",{})
if integrity:add(f"Аудит комплектности: записей — {len(integrity.get('recordings',[]))}; ошибок — {len(integrity.get('errors',[]))}.")
story.append(PageBreak())
heading("Источники")
for ref in [
 "[1] Task Force of ESC and NASPE. Heart rate variability. Circulation, 1996, 93:1043–1065. https://doi.org/10.1161/01.CIR.93.5.1043",
 "[2] Wang W. et al. Algorithmic Principles of Remote PPG. IEEE TBME, 2017, 64(7):1479–1491. https://doi.org/10.1109/TBME.2016.2609282",
 "[3] Scikit-learn. Grouped cross-validation, LeaveOneGroupOut. https://scikit-learn.org/stable/modules/cross_validation.html",
 "[4] SciPy. scipy.stats.permutation_test. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html",
 "[5] Google MediaPipe. Face Landmarker and Pose Landmarker. https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker",
 "[6] SYSTRAN. faster-whisper. https://github.com/SYSTRAN/faster-whisper",
]:add(ref,"SmallRu")
heading("Файлы результатов")
table(["Путь в проекте","Содержимое"],[
 ["outputs/research/content_comparison.csv","Сопоставления с цитатами, контекстом и временем"],
 ["outputs/audio/","Стенограммы, временные метки, акустические признаки"],
 ["outputs/bio/","ЭКГ, ФПГ, RIP, GSR, интервалы и показатели ВСР"],
 ["outputs/video/","Лицевые области, rPPG и кинематика"],
 ["outputs/sync/","Фоновые участки и оценки смещения"],
 ["data/private/","Письменные эталоны и каталог источников"],
 ["docs/validation.md","Команды и результаты технических проверок"],
],[235,260])

def footer(c,doc):
    c.setFont("Body",8);c.setFillColor(colors.HexColor("#607071"))
    c.drawRightString(A4[0]-50,28,str(doc.page))

path=args.output
path.parent.mkdir(parents=True,exist_ok=True)
doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=42,bottomMargin=45,
                     title="Лабораторная работа НТИ. Анализ видео, речи и биосигналов",author="")
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(path)
