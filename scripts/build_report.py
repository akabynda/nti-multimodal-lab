"""Build the local Russian laboratory report from calculated outputs.

Uses reportlab and an available Cyrillic TrueType font; private results are never
embedded into the repository. Run after processing and validation.
"""
from pathlib import Path
import argparse
import hashlib
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
audio_recheck_path=OUT/"research/audio_recheck.json"
audio_recheck=read(audio_recheck_path,{})
report_addenda=read(ROOT/"data/private/report_recheck_notes.json",{})
if audio_recheck:
    current_comparison_sha=hashlib.sha256((OUT/"research/content_comparison.json").read_bytes()).hexdigest()
    if audio_recheck.get("source_comparisons_sha256")!=current_comparison_sha:
        raise ValueError("audio_recheck refers to a different content_comparison.json; refresh the recheck before reporting")
    if report_addenda and (report_addenda.get("source_comparisons_sha256")!=current_comparison_sha
                          or report_addenda.get("source_audio_recheck_sha256")!=hashlib.sha256(audio_recheck_path.read_bytes()).hexdigest()):
        raise ValueError("Report context addenda do not match the source comparison/recheck hashes")
recheck_by_id={row["comparison_id"]:row for row in audio_recheck.get("candidate_checks",[])}
sources=read(ROOT/"data/private/stories.json",{}).get("stories",[])
audios=[read(p) for p in sorted((OUT/"audio").glob("*.segments.json"))]
videos=[read(p) for p in sorted((OUT/"video").glob("*.json"))]
alignments=[read(p) for p in sorted((OUT/"sync").glob("*.json")) if p.stem!="summary"]
by_bio={x["subject_id"]:x for x in bio}
by_video={x["participant_id"]:x for x in videos}
order=[a["participant_id"] for a in audios]
# Personal display names are supplied locally, never embedded in public code.
private_names=read(ROOT/"data/private/participant_names.json",{})
names={pid:private_names.get(pid,f"Участник P{i:02}") for i,pid in enumerate(order,1)}
total_windows=sum(b.get("window_count",0) for b in bio)
by_alignment={a.get("participant_id",a.get("subject_id")):a for a in alignments}
gsr_review_count=sum(b.get("gsr_quality",{}).get("status")=="flat_or_out_of_range_review_required" for b in bio)
relation={"consistent":"Согласуется","different":"Другой пример или деталь","omitted":"Не рассказано","uncertain":"Неоднозначно","contradiction_candidate":"Кандидат на противоречие"}
counts=Counter(n["relation"] for n in notes)

add("Мультимодальное сопоставление рассказов и физиологических сигналов","ReportTitle")
add("Лабораторная работа НТИ","Section")
add(f"Отчёт по предоставленным записям и правдивым письменным историям\n{REPORT_DATE}")
heading("Основной результат")
add(f"Подготовлен локальный программный комплекс. Доступны {len(audios)} результатов ASR, {len(bio)} результатов БиоРадио и {len(videos)} результатов видеоаналитики. Выполнена содержательная сверка {review.get('stories_reviewed',0)} письменных историй с автоматическими стенограммами. Все письменные тексты приняты как правдивый эталон по прямому подтверждению владельца данных.")
add(f"Содержательная сверка содержит {len(notes)} записи. По категориям: согласуется — {counts['consistent']}; кандидат на противоречие — {counts['contradiction_candidate']}; другой пример или деталь — {counts['different']}; эпизод не рассказан — {counts['omitted']}; неоднозначно — {counts['uncertain']}. Кандидаты относятся к {review.get('stories_with_contradiction_candidates',0)} исходным историям. Это число замечаний, а не число независимых случаев лжи.")
add("Для каждого сопоставления доступны исходная цитата, цитата ASR, контекст вопроса и ответа, время на видео и объяснение. Прямые расхождения выделены для последующего прослушивания. Ошибка распознавания речи, непересказанная история и другой пример не считаются доказанной ложью.")
heading("Граница полученных выводов")
add("Система измеряет физиологию и движения, но не устанавливает намерение человека по внешности или биосигналам. Модель правдивости и влияние классического темперамента нельзя валидировать на данном наборе без независимой разметки ответов и измерений темперамента. В отчёте отсутствующие результаты обозначены как недоступные, без подстановки искусственных данных.")
table(["Материал","Фактический объём"],[
    ["Видео и речь",f"{len(audios)} записей; {sum(a.get('duration_s',0) for a in audios)/60:.2f} мин аудио; {sum(len(a.get('segments',[])) for a in audios)} сегментов ASR"],
    ["БиоРадио",f"{len(bio)} записей; каналы ECG, RIP, GSR; частоты сохранены в metadata.json"],
    ["Письменные истории",f"{research.get('source_documents',0)} документов; {len(sources)} правдивых историй"],
    ["Камеры","Один ракурс; второй исключён по указанию владельца"],
    ["ВСР",f"{total_windows} перекрывающихся окон по 120 с с шагом 5 с"],
],[155,340])
story.append(PageBreak())
heading("Набор данных и соответствие записей")
add("Имена переименованных видео связаны с файлами БиоРадио и документами по указанию владельца. Распознавание личности по лицу или голосу не применяется. Время начала видео не равно времени запуска прибора.")
rows=[]
for i,a in enumerate(audios,1):
    b=by_bio.get(a["participant_id"],{})
    media_duration=by_alignment.get(a["participant_id"],{}).get("media_duration_s",a.get("duration_s"))
    difference=(b["analysis_duration_s"]-media_duration) if b.get("analysis_duration_s") is not None and media_duration is not None else None
    rows.append([f"P{i:02}",names.get(a["participant_id"],a["participant_id"]),num(media_duration),
                 num(b.get("analysis_duration_s")),num(difference)])
table(["ID","Участник","Медиа, с","БОС, с","Разница, с"],rows,[35,190,90,90,90])
add("Длительность медиа берётся из результата синхронизации, а при его отсутствии — из декодированной аудиодорожки.")
add("Разница длительностей ограничивает сумму раннего запуска и поздней остановки БиоРадио. Она не определяет каждую часть по отдельности. Псевдонимы P01–P08 используются на общих графиках в порядке видео.")
heading("Фактические каналы")
add("В BCRX включены ЭКГ, дыхательный пояс RIP и кожная проводимость GSR. Контактный ФПГ отсутствует. Следовательно, сравнение контактного ФПГ и ЭКГ из задания не может быть выполнено на этих исходниках. Поддержка расчёта пульсовых интервалов PPG реализована и проверена на тестовом сигнале; это не замена отсутствующей записи.")
add(f"Проверку плоских участков или значений вне диапазона GSR требуют {gsr_review_count} из {len(bio)} сессий. GSR сохраняется для аудита, но не используется как достоверный индекс стресса. Счётчики пропущенных пакетов и NaN сохранены в метаданных; пропуски не заменяются нулями.")
heading("Профили")
add("Таблица содержит 34 строки профилей. Только два участника связаны с ней по явно подтверждённым табельным номерам. Типы «Содействующий», «Контролирующий» и другие не эквивалентны автоматически холерику, сангвинику, флегматику и меланхолику. Классический темперамент всех восьми участников остаётся неизвестным.")
story.append(PageBreak())
heading("Синхронизация модальностей")
add("Каноническое начало сессии — первый декодированный видеокадр. Для привязки аудиодорожки используются времена первых декодированных аудио- и видеокадров. Заголовок stream.start_time может отличаться из-за порядка кодирования и не заменяет эту проверку.")
origins=[a.get("media_timebase",{}) for a in alignments]
decoded_origins=[m for m in origins if m.get("media_origin_status") is None
                 and m.get("audio_first_decoded_pts_s") is not None and m.get("video_first_decoded_pts_s") is not None]
if decoded_origins:
    errors=[abs(m["audio_first_decoded_pts_s"]-m["video_first_decoded_pts_s"]) for m in decoded_origins]
    add(f"В имеющихся результатах проверено {len(decoded_origins)} исходных файлов. Максимальная разница начал аудио и видео: {num(max(errors)*1000,2)} мс. Параметры и происхождение временных шкал сохранены в media_timebase.")
add("Владелец подтвердил порядок: БиоРадио запускается раньше камеры, камера останавливается раньше БиоРадио. Если Lb и Lv — длительности БОС и видео, допустимое опережение прибора находится в диапазоне [0; Lb − Lv]. Часы задаются формулой tсессии = offset + scale × tБОС.")
add("Кандидаты фоновых фрагментов в начале и конце выделяются по инструкциям в ASR; при отсутствии инструкции используется явно отмеченная временная эвристика. Геометрия глаз даёт приблизительную дополнительную проверку, но не доказывает выполнение инструкции или отсутствие движения. Закрытие глаз непосредственно не измеряется ЭКГ или RIP и не служит самостоятельной точной меткой прибора.")
add("Внутри спокойных фрагментов сравниваются пульсовая компонента rPPG и сформированный по R-зубцам опорный пульсовой ряд. Проверяются оба конца записи, альтернативные пики корреляции и качество сигнала. Неизвестное время распространения пульсовой волны учитывается как неопределённость. По неоднозначным корреляциям точная синхронизация не объявляется.")
if alignments:
    rows=[]
    statuses={"supported_unverified":"Кандидат поддержан, не проверен",
              "ambiguous_unverified":"Неоднозначно, не проверено",
              "insufficient_signal":"Недостаточно сигнала"}
    for pid in order:
        a=by_alignment.get(pid,{})
        status=a.get("status","Результат отсутствует")
        rows.append([names.get(pid,pid),statuses.get(status,status),interval(a.get("lead_bounds_s")),
                     num(a.get("bio_to_video_offset_s",(a.get("candidate") or {}).get("bio_to_video_offset_s")),2)])
    table(["Участник","Статус","Опережение, с","Кандидат offset, с"],rows,[135,140,110,110])
    add("Автоматический кандидат и статус поддержки не подтверждают синхронизацию. Смещение нельзя считать проверенным без независимого события или ручной проверки. Неопределённость и альтернативные пики сохранены в outputs/sync.")
else:
    add("Подробные результаты подбора смещения доступны в outputs/sync после завершения обработки видео. Ручной offset и масштаб часов поддерживаются интерфейсом.")
add("Без независимых меток дрейф часов не выдаётся за измеренный: scale остаётся 1. По двум проверенным опорным событиям модуль sync.py может оценить масштаб и остаточную ошибку.")
story.append(PageBreak())
heading("Обработка ЭКГ и дыхания")
add("Контейнер BCRX прочитан напрямую: ZIP, внутренние блоки raw DEFLATE, little-endian float32. Частоты, единицы и коэффициенты берутся из header.xml. Число отсчётов сверено с длительностью. Автоматическая детекция R-зубцов сопровождается отбраковкой неправдоподобных и локально аномальных RR-интервалов.")
add("В каждом окне 120 с рассчитаны ЧСС, SDNN, RMSSD, pNN50 и спектральные показатели. При недостаточном покрытии или длинных разрывах спектр не публикуется как валидный. ULF оставлен неопределённым; VLF и общая мощность для двух минут имеют исследовательский статус. LF/HF не интерпретируется как вероятность стресса или лжи [1].")
table(["Участник","Окон","Прошли QC","Медиана ЧСС"],[
 [f"P{i+1:02}",str(by_bio.get(pid,{}).get("window_count","")),str(by_bio.get(pid,{}).get("usable_ecg_windows","")),num(by_bio.get(pid,{}).get("median_hr_bpm"))]
 for i,pid in enumerate(order)],[80,110,145,160])
fig=OUT/"bio/aggregate_summary.png"
if fig.exists():story.append(Image(str(fig),width=495,height=207))
add("Рисунок 1. Автоматическое качество ЭКГ и описательные показатели. Перекрывающиеся окна не являются независимыми наблюдениями. Результаты требуют проверки R-зубцов на участках с артефактами.","SmallRu")
add("RIP даёт прямую дыхательную кривую пояса, EDR — дополнительную оценку по амплитудам ЭКГ. Частота дыхания показывается только при достаточной периодичности. EDR не используется как независимый эталон для собственной ЭКГ.")
story.append(PageBreak())
heading("Видеоаналитика")
add("Записи имеют размер 1440×1080 с неквадратным пикселем SAR 4:3 и отображением 16:9. Для малого лица в кадре сначала выделяется область головы по точкам позы, затем применяется Face Landmarker. Цвета берутся из исходных пикселей лба и щёк. Рот и глаза отслеживаются отдельно, но не используются как основной источник пульса.")
add("POS-rPPG рассчитывается в скользящих окнах с подавлением общей яркостной компоненты [2]. После полосовой фильтрации 0,7–3,5 Гц оценивается спектральная частота в 30-секундных окнах с шагом 5 с. Пропуски обнаружения лица не соединяются через длинные интервалы. Спектральная концентрация служит индикатором качества, а не доказательством точного пульса.")
table(["Участник","Кадры с ROI","Частота анализа","Окон всего / с ЧСС"],[
 [f"P{i+1:02}",percent(by_video.get(pid,{}).get("face_coverage")),num(by_video.get(pid,{}).get("analysis_fps"))+" Гц",
  f"{len(by_video[pid].get('windows',[]))} / {sum(w.get('rppg_hr_bpm') is not None for w in by_video[pid].get('windows',[]))}" if pid in by_video else "н/д"]
 for i,pid in enumerate(order)],[90,140,140,125])
video_quality_figure=OUT/"audit/video_quality.png"
if video_quality_figure.exists():
    story.append(Image(str(video_quality_figure),width=495,height=264))
    add("Рисунок 2. Доступность лицевых ROI в кадрах и доля 30-секундных окон, где частота rPPG прошла автоматический спектральный фильтр. Это разные знаменатели и показатели доступности, а не точность определения пульса. Проверка по ЭКГ требует подтверждённого смещения; перекрывающиеся окна зависимы.","SmallRu")
add("Положение головы, плеч, кистей, таза и стоп даёт двумерные кинематические признаки. Отсутствующие или скрытые точки исключаются. Угол оси плечи–таз — приближённый наклон корпуса; он не равен анатомической кривизне позвоночника или настоящему центру тяжести. Колебания плеч — дыхательный двигательный прокси, а не измерение объёма грудной клетки.")
heading("Речь и акустика")
add("Речь распознана локально faster-whisper large-v3-turbo с временными метками слов. Извлечены основной тон, его изменчивость, энергия, доля звучащей речи, паузы, спектральный центр, наклон и форманты. Число слогов оценивается по гласным русского текста и остаётся приближённым. Временные границы ограничены длительностью записи.")
add("Разделение голосов выполняется по нейросетевым речевым представлениям WeSpeaker. Названия кластеров не равны ролям. Роли интервьюера и респондента назначаются после прослушивания; без этого переход между голосами не считается проверенной задержкой ответа. Длинные ответы и короткие реплики требуют отдельной проверки.")
heading("Правдивый эталон и содержание ответов")
add("Единица сравнения — конкретное утверждение, связанное с письменной историей. Сначала автоматический поиск TF-IDF предлагает подходящий участок. Затем выполняется содержательная сверка события, лица, времени, действия и отрицания. Проверяется соседний вопрос: реплика интервьюера не должна становиться признанием респондента.")
table(["Категория","Количество","Смысл"],[
 [relation[k],str(counts[k]),v] for k,v in [
 ("consistent","Устная деталь согласуется с известным правдивым текстом."),
 ("contradiction_candidate","В ASR обнаружено потенциально несовместимое утверждение; нужна проверка звука и контекста."),
 ("different","Рассказан другой пример либо добавлена не проверяемая исходником деталь."),
 ("omitted","Письменный эпизод не был пересказан в интервью."),
 ("uncertain","Недостаточно ясный ответ или ненадёжная интерпретация ASR.")]],
 [145,70,280])
add(f"Все цитаты проверены на наличие в исходном тексте и соответствующем интервале ASR. Это проверка трассируемости, а не подтверждение правильности распознавания на слух. Полная таблица content_comparison.csv содержит {len(notes)} строк, включая ограничения и статус audio_verified. В интерфейсе эпизод открывается одним нажатием.")
add("Сильные кандидаты здесь основаны на прямых отрицаниях известных событий. Более слабые случаи, например «не помню», «обычно не пью», смена планов или ответ о детстве, описываются с оговорками. Намерение обмануть не следует автоматически из любой разницы между двумя рассказами.")

if audio_recheck:
    story.append(PageBreak())
    heading("Повторная автоматическая проверка спорных фрагментов")
    recheck_summary=audio_recheck.get("summary",{})
    add(f"Из исходных аудиодорожек повторно распознано {recheck_summary.get('unique_original_audio_clips',len(audio_recheck.get('clips',[])))} отдельных фрагментов. Они связаны с {recheck_summary.get('candidate_records_checked',0)} заметками и {recheck_summary.get('unique_spoken_assertions',0)} различными устными утверждениями. Использована полная модель {audio_recheck.get('primary_model','н/д')} вместо исходной turbo-версии, с двумя вариантами декодирования.")
    add(f"Перепроверка охватывает выбранные {recheck_summary.get('candidate_records_checked',0)} из {counts['contradiction_candidate']} предварительных кандидатов; она не является повторной проверкой всех ответов или всех письменных историй.")
    table(["Показатель","Результат"],[
        ["Уточнения цитат",str(recheck_summary.get("quote_refinements",0))],
        ["Изменения полярности целевого утверждения",str(recheck_summary.get("polarity_reversed_records",0))],
        ["Неуверенная формулировка вопроса",str(recheck_summary.get("question_wording_uncertain_records",0))],
        ["Подтверждено прослушиванием человеком",str(recheck_summary.get("human_audio_verified_records",0))],
        ["Требуют экспертной проверки",str(recheck_summary.get("requires_expert_review_records",0))],
    ],[365,130])
    add("Повторные проходы автоматически воспроизвели целевые отрицания. Это подтверждение устойчивости распознанной формулировки в пределах моделей Whisper, а не подтверждение лжи или правильного определения говорящего. Два декодирования одной модели и исходная модель того же семейства не являются независимой проверкой человеком. audio_verified остаётся false.")
    refinements=[r for r in audio_recheck.get("candidate_checks",[]) if r.get("quote_change") not in (None,"unchanged")]
    for r in refinements:
        add(f"{r['comparison_id']} · {timecode(r.get('candidate_start_s'))}–{timecode(r.get('candidate_end_s'))}","Sub")
        add(f"Исходная ASR: «{r.get('original_spoken_excerpt','')}»\nУстойчивая цитата повторной ASR: «{r.get('recommended_stable_excerpt','')}»","SmallRu")
        add(r.get("assessment_note",""),"SmallRu")
    for addition in report_addenda.get("addenda",[]):
        add(addition.get("heading","Контекст, влияющий на интерпретацию"),"Sub")
        add(f"{addition['comparison_id']} · {timecode(addition.get('start_s'))}–{timecode(addition.get('end_s'))}","SmallRu")
        add(f"Следующий вопрос по повторной ASR: «{addition.get('question_excerpt','')}»\nОтвет: «{addition.get('spoken_excerpt','')}» ({timecode(addition.get('answer_start_s'))}–{timecode(addition.get('answer_end_s'))}).","SmallRu")
        add(addition.get("interpretation",""),"SmallRu")
    add("Исходная таблица content_comparison.json не перезаписывается. Дополнительные результаты и уточнения сохраняются отдельно; категории не переобозначаются в ложь. SHA-256 исходной таблицы: "+audio_recheck["source_comparisons_sha256"],"SmallRu")

for pid in order:
    story.append(PageBreak())
    heading(names.get(pid,pid))
    summ=next((s for s in review.get("participant_summaries",[]) if s["participant_id"]==pid),{})
    add(summ.get("content_summary") or summ.get("summary","Результаты сверки приведены ниже."))
    for addition in report_addenda.get("addenda",[]):
        if addition.get("participant_id")==pid:
            add(addition.get("participant_addendum",addition.get("interpretation","")))
    rows=[]
    for src in [s for s in sources if s["participant_id"]==pid]:
        n=[n for n in notes if n["story_id"]==src["story_id"]]
        cats=list(dict.fromkeys(relation.get(x["relation"],x["relation"]) for x in n))
        rows.append([src["title"],"; ".join(cats)])
    table(["Письменная история","Результат сверки"],rows,[240,255])
    rank={"contradiction_candidate":0,"uncertain":1,"different":2,"consistent":3,"omitted":4}
    chosen=sorted([n for n in notes if n["participant_id"]==pid and n.get("spoken_excerpt")],
                  key=lambda n:(rank.get(n["relation"],9),-float(n.get("confidence",0))))[:3]
    for n in chosen:
        add(f"{relation.get(n['relation'],n['relation'])}  {timecode(n.get('start_s'))}–{timecode(n.get('end_s'))}","Sub")
        add(f"Эталон: «{trim(n.get('source_excerpt'),210)}»","SmallRu")
        add(f"Речь по ASR: «{trim(n.get('spoken_excerpt'),210)}»","SmallRu")
        repeated=recheck_by_id.get(n["comparison_id"])
        if repeated and repeated.get("quote_change") not in (None,"unchanged"):
            add(f"Уточнение повторной автоматической ASR: «{trim(repeated.get('recommended_stable_excerpt'),210)}». Исходная цитата сохранена для аудита.","SmallRu")
        if n.get("question_context"):
            add(f"Вопрос по ASR: «{trim(n['question_context'],250)}»","SmallRu")
        elif n.get("context_excerpt") and len(n.get("spoken_excerpt","").split())<5:
            add(f"Контекст по ASR: «{trim(n['context_excerpt'],350)}»","SmallRu")
        add(trim(n.get("interpretation"),400),"SmallRu")
    add("Цитаты сокращены только в этом обзорном отчёте. Полный текст, окружающие реплики и границы эпизода сохранены в таблице и приложении. Эти категории описывают сведения, а не личность участника.","SmallRu")

story.append(PageBreak())
heading("Статистические гипотезы и модель")
add("H1: индивидуальный контраст признака между проверенными правдивыми и ложными ответами не зависит от измеренного классического темперамента. H2: добавление темперамента улучшает прогноз на новом участнике. Эти гипотезы сформулированы как план исследования; значимость и точность на данном наборе не заявляются.")
add("Реализован перестановочный анализ индивидуальных контрастов, поправка Холма, оценка эффекта и bootstrap по участникам. Для модели предусмотрена логистическая регрессия с leave-one-participant-out. Импутация, масштабирование и кодирование обучаются только на тренировочной части. Сравниваются одинаковые варианты без темперамента и с ним [3, 4].")
add(f"Случайное разделение {total_windows} перекрывающихся окон создало бы риск утечки информации одного человека и завышенной точности. {len(bio)} участников не превращаются в {total_windows} независимых наблюдений. Ни отсутствие подтверждённого эффекта, ни отказ вычислить p не доказывают отсутствия эффекта.")
table(["Необходимое условие","Состояние"],[
 ["Правдивые письменные источники","Подтверждены пользователем для всех 48 историй"],
 ["Проверенные метки отдельных устных ответов","Не предоставлены; содержательные кандидаты требуют прослушивания"],
 ["Классический темперамент","Не измерен; другая типология профилей не подменяется"],
 ["Точная межмодальная синхронизация","Оценивается отдельно с неопределённостью; ручная проверка поддерживается"],
 ["Независимая оценка модели","Код реализован; численную точность по реальным меткам пока вычислять некорректно"],
],[220,275])
heading("Что требуется для завершённого эксперимента")
add("Прослушать отмеченные ответы, независимо разметить экспериментальное условие и проверить межэкспертное согласие; подтвердить синхронизацию; получить валидные измерения темперамента и увеличить число участников. После этого использовать фиксированный набор признаков и групповой протокол проверки, не подбирая правила по итоговому тесту.")
story.append(PageBreak())
heading("Интерфейс и воспроизводимость")
add("Веб-интерфейс работает локально на FastAPI. Пять зон связаны общей шкалой: навигация по времени; речь и правдивые истории; ЭКГ, RIP, GSR и rPPG; один видеопоток с ROI и точками позы; спектр ВСР и текущие численные признаки. Настройка смещения и разметка сохраняются в JSON.")
interface_path=OUT/"report/interface.png"
if interface_path.exists():
    interface_image=Image(str(interface_path))
    image_scale=min(495/interface_image.imageWidth,540/interface_image.imageHeight)
    interface_image.drawWidth=interface_image.imageWidth*image_scale
    interface_image.drawHeight=interface_image.imageHeight*image_scale
    story.append(interface_image)
    add("Рисунок 3. Фактический локальный интерфейс с видео, ROI, позой, сигналами и показателями. Снимок содержит приватные материалы исследования и включается только в локальный отчёт; он исключён из публичного репозитория. Видимость данных не означает подтверждение синхронизации или достоверности отдельных высказываний.","SmallRu")
add("Для запуска в подготовленной папке используется start.cmd. Установка с нуля, Windows/Linux/macOS, CPU/GPU, зависимости и команды обработки описаны в README.md. Подготовлены Dockerfile, docker-compose.yml, environment.yml, модульные тесты и GitHub Actions. Исходные данные и персональные результаты исключены из публикации.")
integrity=read(OUT/"audit/project.json",{})
if integrity:
    add(f"Автоматический аудит целостности: {integrity.get('status','н/д')}; проверено записей: {len(integrity.get('recordings',[]))}; замечаний: {len(integrity.get('errors',[]))}. Он проверяет комплектность и форматы, но не заменяет проверку точности алгоритмов по независимой разметке.")
validation=read(OUT/"validation.json",{})
if validation:
    table(["Проверка","Результат"],[[k,str(v)] for k,v in validation.items()],[210,285])
else:
    add("Подробные команды и результаты проверки приведены в docs/validation.md.")
heading("Соответствие заданию")
table(["Блок","Реализовано и ограничения"],[
 ["Синхронизация","Явные часы, ручные метки, оценка смещения в допустимом диапазоне; неоднозначность показывается"],
 ["Видео","Лицевые ROI, POS-rPPG, видимая поза и кинематика, дыхательное движение; один ракурс"],
 ["Аудио и БОС","Полная русская ASR, голосовые признаки, диаризация, ЭКГ и RIP; контактный ФПГ отсутствует"],
 ["Интерфейс","Пять синхронных зон, покадровая навигация, эталоны и доказательные эпизоды"],
 ["Исследование","48 историй сверены; статистический и модельный протокол реализован; недостающие метки и темперамент явно отмечены"],
 ["Воспроизводимость","Исходный код, тесты, контейнер, настройки, ноутбук и локальные рассчитанные результаты"],
],[130,365])
story.append(PageBreak())
heading("Источники и файлы результатов")
refs=[
 "[1] Task Force of the European Society of Cardiology and NASPE. Heart rate variability. Standards of measurement, physiological interpretation, and clinical use. Circulation, 1996, 93:1043–1065. https://doi.org/10.1161/01.CIR.93.5.1043",
 "[2] Wang W., den Brinker A.C., Stuijk S., de Haan G. Algorithmic Principles of Remote PPG. IEEE TBME, 2017, 64(7):1479–1491. https://doi.org/10.1109/TBME.2016.2609282",
 "[3] Scikit-learn. Cross-validation: grouped data and LeaveOneGroupOut. https://scikit-learn.org/stable/modules/cross_validation.html",
 "[4] SciPy. scipy.stats.permutation_test. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html",
 "[5] Google MediaPipe. Face Landmarker and Pose Landmarker. https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker",
 "[6] SYSTRAN. faster-whisper. https://github.com/SYSTRAN/faster-whisper",
 "[7] BioRadio manufacturer. Device documentation and BioCapture specifications; описание фактически декодированного формата — docs/bioradio_format.md.",
]
for r in refs:add(r,"SmallRu")
heading("Основные производные результаты")
for text in [
 "outputs/research/content_comparison.csv — полная содержательная сверка с цитатами и контекстом.",
 "outputs/research/content_comparison_report.json — сводки по 48 историям и восьми участникам.",
 "outputs/research/audio_recheck.json — повторная автоматическая ASR выбранных фрагментов, хеши и ограничения проверки.",
 "outputs/audio/*.segments.json и *.txt — стенограммы; *.features.csv — акустика.",
 "outputs/bio/*/metrics.csv — показатели окон ВСР; signals.npz — сигналы и RR.",
 "outputs/video/*.json — rPPG, кинематика, ROI и индикаторы качества.",
 "outputs/sync/ — кандидаты синхронизации и неопределённость.",
 "data/private/ — эталонные источники, происхождение меток и ручная разметка.",
]:add(text,"SmallRu")
add("Ссылки на исходные личные документы сохранены в локальном каталоге историй. Публичный репозиторий содержит программный код и общую методику; личные тексты, видео, биосигналы и индивидуальные выводы в него не включаются.")

def footer(c,doc):
    c.setFont("Body",8);c.setFillColor(colors.HexColor("#607071"))
    c.drawString(50,28,f"НТИ  •  Мультимодальная лаборатория  •  {REPORT_DATE}")
    c.drawRightString(A4[0]-50,28,str(doc.page))

path=args.output
path.parent.mkdir(parents=True,exist_ok=True)
doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=42,bottomMargin=45,
                     title="Мультимодальное сопоставление рассказов и физиологических сигналов",author="Лабораторная работа НТИ")
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(path)
