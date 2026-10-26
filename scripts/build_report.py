"""Build the local Russian laboratory report from calculated outputs.

Uses reportlab and an available Cyrillic TrueType font; private results are never
embedded into the repository. Run after processing and validation.
"""
from pathlib import Path
import argparse
import json
import csv
import math
from collections import Counter
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, PageBreak, Image, KeepTogether)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from nti_lab.palette import BLUE, INK, MUTED, LINE, SURFACE, WHITE
from nti_lab.story_topics import story_topic_title
from nti_lab.multimodal_analysis import TYPE_CORRECTIONS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output",type=Path,default=OUT/"report"/"Отчет НТИ.pdf")
parser.add_argument("--repository",default="https://github.com/akabynda/nti-multimodal-lab")
args=parser.parse_args()

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
    styles[name].textColor=colors.HexColor(INK)
styles.add(ParagraphStyle("CoverOrg",fontName="BodyBold",fontSize=13,leading=18,alignment=TA_CENTER))
styles.add(ParagraphStyle("ReportTitle",fontName="BodyBold",fontSize=24,leading=30,spaceAfter=12,alignment=TA_CENTER,textColor=colors.HexColor(BLUE)))
styles.add(ParagraphStyle("CoverSubtitle",fontName="BodyBold",fontSize=15,leading=20,alignment=TA_CENTER,textColor=colors.HexColor(BLUE)))
styles.add(ParagraphStyle("CoverText",fontName="Body",fontSize=11,leading=16,alignment=TA_CENTER))
styles.add(ParagraphStyle("Section",fontName="BodyBold",fontSize=15,leading=20,spaceBefore=14,spaceAfter=10,keepWithNext=True,textColor=colors.HexColor(BLUE)))
styles.add(ParagraphStyle("Sub",fontName="BodyBold",fontSize=11,leading=15,spaceBefore=10,spaceAfter=5,keepWithNext=True))
styles.add(ParagraphStyle("TextRu",fontName="Body",fontSize=10,leading=14,spaceAfter=8))
styles.add(ParagraphStyle("SmallRu",fontName="Body",fontSize=8.4,leading=11,spaceAfter=6))
styles.add(ParagraphStyle("CellRu",fontName="Body",fontSize=8.2,leading=11,spaceAfter=0))
styles.add(ParagraphStyle("HeaderCellRu",parent=styles["CellRu"],fontName="BodyBold",textColor=colors.HexColor(WHITE)))
styles.add(ParagraphStyle("TableCaption",fontName="Body",fontSize=9,leading=12,spaceBefore=4,spaceAfter=4,keepWithNext=True))
styles.add(ParagraphStyle("FigureCaption",fontName="Body",fontSize=9,leading=12,spaceBefore=4,spaceAfter=10,alignment=TA_CENTER))
story=[]
numbers={"table":0,"figure":0}
def p(text,style="TextRu"):
    return Paragraph(escape(str(text)).replace("\n","<br/>"),styles[style])
def add(text,style="TextRu"):
    story.append(p(text,style))
def heading(text):
    add(text,"Section")
def link(url):
    return f'<a href="{escape(url)}" color="{BLUE}">{escape(url)}</a>'
def table(caption,headers,rows,widths):
    numbers["table"]+=1
    add(f"Таблица {numbers['table']} – {caption}","TableCaption")
    cells=[[p(x,"HeaderCellRu") for x in headers]]+[[p(x,"CellRu") for x in row] for row in rows]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor(BLUE)),
        ("FONTNAME",(0,0),(-1,0),"BodyBold"),
        ("GRID",(0,0),(-1,-1),.4,colors.HexColor(LINE)),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor(SURFACE)]),
    ]))
    t.spaceAfter=10;story.append(t)
    return numbers["table"]
def figure(path,caption,max_height=None):
    if not path.exists():return
    im=Image(str(path));scale=495/im.imageWidth
    if max_height:scale=min(scale,max_height/im.imageHeight)
    im.drawWidth=im.imageWidth*scale;im.drawHeight=im.imageHeight*scale
    numbers["figure"]+=1
    story.append(KeepTogether([im,p(f"Рисунок {numbers['figure']} – {caption}","FigureCaption")]))
def model_name(name):
    return name.replace("BioRadio","БиоРадио").replace("анкета","профили")
def timecode(t):
    if t is None:return "нет сопоставимого интервала"
    return f"{int(t//60):02d}:{t%60:05.2f}"
def trim(t,n=240):
    t=str(t or "")
    return t if len(t)<=n else t[:n].rsplit(" ",1)[0]+"…"
def decimal(text):
    return text.replace(".",",").replace("-","−")
def num(v,d=1):
    try:return decimal(f"{float(v):.{d}f}") if math.isfinite(float(v)) else "н/д"
    except (TypeError,ValueError):return "н/д"
def plain(v):
    try:return decimal(f"{float(v):g}") if math.isfinite(float(v)) else "н/д"
    except (TypeError,ValueError):return "н/д"
def plural(n,one,few,many):
    n=abs(int(n))%100
    return one if n%10==1 and n!=11 else few if 2<=n%10<=4 and not 12<=n<=14 else many
def counted(n,one,few,many):
    return f"{n} {plural(n,one,few,many)}"
def interval(value):
    return "–".join(num(x,2) for x in value) if isinstance(value,list) and len(value)==2 else "н/д"
def percent(value):
    return num(float(value)*100)+" %" if value is not None else "н/д"

bio=read(OUT/"bio/summary.json",{}).get("recordings",[])
research=read(OUT/"research/summary.json",{})
multimodal=read(OUT/"research/multimodal/summary.json",{})
review=read(OUT/"research/content_comparison_report.json",{})
notes=read(OUT/"research/content_comparison.json",{}).get("comparisons",[])
report_edits=read(ROOT/"data/private/report_editorial.json",{}).get("comparisons",{})
notes=[{**note,**{key:value for key,value in report_edits.get(note["comparison_id"],{}).items()
                 if key in {"question_summary","source_excerpt","spoken_excerpt"}}} for note in notes]
label_review=read(OUT/"research/multimodal/label_review.json",{})
trial_by_comparison={cid:trial for trial in label_review.get("strict_trials",[])
                     for cid in set(trial.get("source_comparison_ids",[])+trial.get("original_reference_comparison_ids",[]))}
sources=read(ROOT/"data/private/stories.json",{}).get("stories",[])
audios=[read(p) for p in sorted((OUT/"audio").glob("*.segments.json"))]
editorial=read(OUT/"audio/editorial_review.json",{})
videos=[read(p) for p in sorted((OUT/"video").glob("*.json"))]
alignments=[read(p) for p in sorted((OUT/"sync").glob("*.json")) if p.stem!="summary"]
by_bio={x["subject_id"]:x for x in bio}
ppg_records=[b for b in bio if b.get("ppg_analysis")]
by_video={x["participant_id"]:x for x in videos}
order=[a["participant_id"] for a in audios]
# Personal display names are supplied locally, never embedded in public code.
private_names=read(ROOT/"data/private/participant_names.json",{})
names={pid:private_names.get(pid,f"Участник P{i:02}") for i,pid in enumerate(order,1)}
by_profile={row["participant_id"]:row for row in csvrows(OUT/"research/profile_link_audit.csv")}
total_windows=sum(b.get("window_count",0) for b in bio)
by_alignment={a.get("participant_id",a.get("subject_id")):a for a in alignments}
gsr_review_count=sum(b.get("gsr_quality",{}).get("status")=="flat_or_out_of_range_review_required" for b in bio)
relation={"consistent":"Согласуется с эталоном","different":"Дополнительные сведения","omitted":"Не рассказано","uncertain":"Неоднозначный ответ","contradiction_candidate":"Несоответствие эталону"}
counts=Counter(n["relation"] for n in notes)
asr_validation=read(OUT/"audio/validation_summary.json",{})

def profile_types(raw):
    tokens=[TYPE_CORRECTIONS.get(t.strip().lower(),t.strip().lower()) for t in str(raw or "").split("/") if t.strip()]
    return "/\n".join(t.capitalize() for t in tokens) or "н/д"
def surname_first(name):
    given,_,family=name.partition(" ")
    return f"{family} {given}" if family else name
authors=sorted([surname_first(private_names[pid]),by_profile.get(pid,{}).get("tab_number") or "н/д"]
               for pid in order if pid in private_names)
add("Университет ИТМО","CoverOrg")
story.append(Spacer(1,200))
add("Лабораторная работа НТИ","ReportTitle")
add("Анализ видео, речи и биосигналов","CoverSubtitle")
story.append(Spacer(1,110))
if authors:
    cover=Table([[p("Выполнили:","Sub"),""]]+[[p(name),p(f"ИСУ {isu}")] for name,isu in authors],
                colWidths=[170,90],hAlign="RIGHT")
    cover.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),0),("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    story.append(cover)
story.append(Spacer(1,140))
story.append(Paragraph("Репозиторий проекта: "+link(args.repository),styles["CoverText"]))
story.append(PageBreak())
heading("Задача")
add("Разработать комплекс синхронного анализа видео, речи и биосигналов и проверить возможность распознавания лжи с учётом профилей участников. В эксперименте участники намеренно изменяли часть ответов. Письменные истории служат правдивым эталоном для разметки.")
add(f"Обработано {len(audios)} записей общей продолжительностью {num(sum(a.get('duration_s',0) for a in audios)/60,2)} мин. Сверено {counted(review.get('stories_reviewed',len(sources)),'история','истории','историй')}. Подготовлены стенограммы с временными метками, показатели речи, ЭКГ и дыхания, видеопульс, кинематика и локальный интерфейс для просмотра результатов.")
table("Результаты сверки устных ответов с письменными историями",["Результат сверки","Число сопоставлений"],[[relation[k],str(counts[k])] for k in relation],[365,130])
add(f"Всего {len(notes)} записей сверки. Одна история может содержать несколько проверяемых утверждений. Для найденных ответов сохранены цитаты, вопрос и время на видео; отсутствующие пересказы отмечены отдельно.")
heading("Материалы")
table("Исходные материалы",["Материал","Объём"],[
 ["Видео",f"{len(videos)} записей; один ракурс"],
 ["Письменные истории",f"{counted(len(sources),'история','истории','историй')}; {counted(research.get('source_documents',0),'исходный документ','исходных документа','исходных документов')}"],
 ["Диалоги",counted(sum(len(a.get('qa_candidates',[])) for a in audios),'пара','пары','пар')+" «вопрос – ответ»"],
 ["БиоРадио",f"{len(bio)} записей; ЭКГ, RIP, GSR; контактный ФПГ в {len(ppg_records)} записи"],
 ["Вариабельность ритма",f"{counted(total_windows,'окно','окна','окон')} по 120 с с шагом 5 с"],
],[155,340])
story.append(PageBreak())
heading("Записи участников")
rows=[]
for i,a in enumerate(audios,1):
    b=by_bio.get(a["participant_id"],{})
    duration=by_alignment.get(a["participant_id"],{}).get("media_duration_s",a.get("duration_s"))
    difference=b.get("analysis_duration_s",0)-duration if duration is not None and b.get("analysis_duration_s") is not None else None
    rows.append([f"P{i:02}",names[a["participant_id"]],num(duration),num(b.get("analysis_duration_s")),num(difference)])
recordings_table=table("Длительность видеозаписей и записей БиоРадио",["ID","Участник","Видео, с","БОС, с","Разница, с"],rows,[35,190,90,90,90])
add(f"Видео, тексты и записи прибора объединены по именам файлов и данным документов. Обозначения P01–P{len(order):02} в таблицах и на рисунках соответствуют порядку таблицы {recordings_table}.")
story.append(PageBreak())
heading("Синхронизация")
add("БиоРадио запускали перед видеозаписью и останавливали после неё. Поэтому длительность БОС больше длительности видео. Разница задаёт допустимый диапазон опережения прибора: от 0 до LБОС − Lвидео.")
add("За начало видео принят первый декодированный кадр. Начальный и конечный фон с закрытыми глазами найдены по речевым инструкциям и геометрии глаз. На каждом фоне видеопульс rPPG сопоставлен с опорным пульсовым рядом, построенным по R-зубцам ЭКГ.")
add("Опережение перебирается по всему допустимому диапазону с шагом не более 0,02 с. Для каждого значения рассчитан модуль коэффициента корреляции r между rPPG и опорным рядом, сдвинутым на это опережение, отдельно на начальном и конечном фоне. Общая оценка — среднее геометрическое двух значений r. Принятое опережение соответствует максимуму общей оценки. Альтернативный максимум — наибольшая общая оценка при опережениях, отличающихся от принятого более чем на 0,2 с. Запас — разность принятого и альтернативного максимумов.")
add(f"Преобразование времени: tвидео = offset + scale × tБОС, где offset = −опережение, scale = 1. Условия статусов приведены в таблице {numbers['table']+1}, результаты — в таблице {numbers['table']+2}.")
statuses={"supported_unverified":"Однозначный максимум","ambiguous_unverified":"Неоднозначный максимум","insufficient_signal":"Мало сигнала"}
table("Условия статусов синхронизации",["Статус","Условие"],[
 [statuses["supported_unverified"],"r ≥ 0,35 на каждом фоне; общая оценка ≥ 0,40; запас общей оценки ≥ 0,04, запас r на каждом фоне ≥ 0,02; максимумы r на двух фонах различаются по опережению не более чем на 0,25 с; ЭКГ на обоих фонах прошла автоматический контроль качества"],
 [statuses["ambiguous_unverified"],"Максимум найден, но хотя бы одно условие однозначного максимума не выполнено"],
 [statuses["insufficient_signal"],"Ни для одного опережения не рассчитаны оба значения r: на фоне менее 75 % отсчётов rPPG, фон короче 40 с или частота кадров rPPG не выше 6 Гц; offset — середина допустимого диапазона"]],[135,360])
sync_table=table("Автоматические оценки смещения БиоРадио относительно видео",["Участник","Статус","Диапазон опережения, с","Общая оценка","Запас","Offset, с"],[
 [names[pid],statuses.get(by_alignment.get(pid,{}).get("status"),"Не рассчитано"),
 interval(by_alignment.get(pid,{}).get("lead_bounds_s")),
 num((by_alignment.get(pid,{}).get("joint_support") or {}).get("best_score"),2),
 num((by_alignment.get(pid,{}).get("joint_support") or {}).get("margin"),3),
 num(by_alignment.get(pid,{}).get("bio_to_video_offset_s",(by_alignment.get(pid,{}).get("candidate") or {}).get("bio_to_video_offset_s")),2)]
 for pid in order],[125,120,85,60,45,60])
story.append(PageBreak())
heading("Данные участников")
profiles_table=table("Профили участников",["Участник","ИСУ","Естественный","Демонстрируемый","Выраженность"],[
 [names[pid],by_profile.get(pid,{}).get("tab_number") or "н/д",
  profile_types(by_profile.get(pid,{}).get("profile_natural_raw")),
  profile_types(by_profile.get(pid,{}).get("profile_demonstrated_raw")),
  plain(by_profile.get(pid,{}).get("expression"))]
 for pid in order],[125,55,110,115,90])
emotions_table=table("Эмоциональные показатели участников",["Участник","Эмоц+","Эмоц−","Эмоц_Баланс"],[
 [f"P{i+1:02}",plain(by_profile.get(pid,{}).get("emotion_positive")),
  plain(by_profile.get(pid,{}).get("emotion_negative")),
  plain(by_profile.get(pid,{}).get("emotion_balance"))]
 for i,pid in enumerate(order)],[75,140,140,140])
for pid in order:
    if by_profile.get(pid,{}).get("profile_link_status")=="requested_id_not_found":
        add(f"{names[pid]}: ИСУ {by_profile[pid]['tab_number']} в исходной таблице отсутствует.","SmallRu")
add(f"Источник таблиц {profiles_table} и {emotions_table}: Результаты.xlsx, сопоставление по номеру ИСУ.","SmallRu")
story.append(PageBreak())
heading("ЭКГ, дыхание и качество сигналов")
add("Файлы BCRX декодированы в численные ряды. Частоты дискретизации, единицы и коэффициенты прочитаны из заголовков. После фильтрации ЭКГ найдены R-зубцы, проверены RR-интервалы и рассчитаны ЧСС, SDNN, RMSSD, pNN50, VLF, LF, HF и LF/HF [1].")
add("Окно ВСР — 120 с, шаг — 5 с. Окна с разрывами или нестабильными RR исключены из итоговых оценок. Диапазоны спектра: VLF 0,0033–0,04 Гц, LF 0,04–0,15 Гц, HF 0,15–0,4 Гц; общая мощность — сумма трёх диапазонов. ULF в окне 120 с не рассчитывается. Дыхательный пояс RIP обработан отдельно; EDR рассчитан по колебаниям амплитуды ЭКГ.")
table("Окна ВСР и контроль качества ЭКГ",["Участник","Окон","Прошли контроль","Медиана ЧСС, уд/мин"],[
 [f"P{i+1:02}",str(by_bio.get(pid,{}).get("window_count","—")),str(by_bio.get(pid,{}).get("usable_ecg_windows","—")),num(by_bio.get(pid,{}).get("median_hr_bpm"))]
 for i,pid in enumerate(order)],[80,110,145,160])
figure(OUT/"bio/aggregate_summary.png","Качество ЭКГ и описательные показатели по участникам")
add(f"В {gsr_review_count} из {len(bio)} записей GSR обнаружены плоские участки или значения вне рабочего диапазона. Эти данные сохранены с отметкой качества.")
if ppg_records:
    story.append(PageBreak())
    heading("Контактный ФПГ и ЭКГ")
    add("Контактный ФПГ записан вместе с ЭКГ одним прибором. Сравнение использует общую шкалу отсчётов БиоРадио. Пульсовые интервалы рассчитаны после фильтрации и контроля качества.")
    table("Сравнение контактного ФПГ и ЭКГ",["Участник","Окон ФПГ / прошли контроль","Сравнено окон","MAE ЧСС, уд/мин"],[
        [names.get(b["subject_id"],b["subject_id"]),
         f"{b['ppg_analysis'].get('window_count','—')} / {b['ppg_analysis'].get('usable_ppg_windows','—')}",
         str(b.get("ecg_ppg_comparison",{}).get("paired_windows","—")),
         num(b.get("ecg_ppg_comparison",{}).get("hr_mae_bpm"),2)] for b in ppg_records],[135,130,110,120])
    for b in ppg_records:
        c=b.get("ecg_ppg_comparison",{})
        add(f"{names.get(b['subject_id'],b['subject_id'])}: средняя разность ЧСС ФПГ − ЭКГ составляет {num(c.get('hr_bias_ppg_minus_ecg_bpm'),2)} уд/мин. Средняя абсолютная разность RMSSD — {num(c.get('rmssd_mae_ms'),2)} мс, SDNN — {num(c.get('sdnn_mae_ms'),2)} мс. В сравнение включены окна, прошедшие контроль обоих каналов.")
        figure(OUT/"bio"/b["subject_id"]/"ecg_ppg_comparison.png","Контактный ФПГ, R-зубцы ЭКГ и сравнение частоты пульса на общей шкале прибора")
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
table("Покрытие лица, видеопульс и дыхательное движение",["Участник","Кадры с ROI","Окна: всего / ЧСС / пары ЭКГ","MAE ЧСС, уд/мин","MAE дыхания, /мин"],rows,[60,85,140,95,115])
add(f"MAE — средняя абсолютная разность частот: видеопульс сопоставлен с ЭКГ, движение плеч — с RIP. Использованы автоматические смещения из таблицы {sync_table} и окна с допустимым качеством; n — число пар дыхательных оценок.","SmallRu")
figure(OUT/"audit/video_quality.png","Доля кадров с лицевыми областями и доля окон с рассчитанной ЧСС rPPG")
story.append(PageBreak())
heading("Распознавание речи и проверка текста")
models_used=", ".join(sorted({str(a.get("model","Whisper")) for a in audios}))
add(f"Расшифровка получена моделью {models_used} через faster-whisper [6]. При вычитке проверены отрицания, короткие ответы и вопросы; сохранены временные метки слов и сегментов.")
if editorial:
    add("Назначены роли интервьюера и респондента, объединены разорванные фразы, исправлены очевидные ошибки текста. Исходные варианты сохранены в журнале правок.")
    table("Редакторская проверка стенограмм",["Проверка","Результат"],[[label,str(editorial[key])] for key,label in [
        ("recordings","Вычитано записей"),("reviewed_segments","Проверено исходных сегментов"),
        ("turns","Реплик после редактирования"),("text_corrections","Текстовых правок"),
        ("merge_groups","Объединено разорванных фраз"),("unresolved_text_fragments","Фрагментов с неуверенным распознаванием")]], [365,130])
elif asr_validation:
    table("Повторная проверка распознавания речи",["Проверка","Результат"],[[label,str(asr_validation[key])] for key,label in [
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
add("По аудиодорожкам извлечены основной тон и его изменчивость, энергия, паузы, доля звучащей речи, спектральный центр, наклон и форманты. Для пар «вопрос – ответ» рассчитаны задержка и длительность ответа, темп в словах/мин и оценка слогов/с по числу гласных. Для начального разделения голосов использованы представления WeSpeaker.")
heading("Сопоставление с эталоном")
add("Письменные истории сопоставлены с интервью по событию, действующему лицу, времени, действию и отрицанию. Короткие ответы рассмотрены вместе с предшествующим вопросом.")
table("Категории сверки с письменным эталоном",["Категория","Критерий"],[[relation[k],v] for k,v in [
 ("consistent","Устный ответ воспроизводит событие или деталь эталона."),
 ("contradiction_candidate","Отрицание факта, замена события, примера или существенной детали эталона."),
 ("different","Сведения вне проверяемого факта; замены эталонного события или детали нет."),
 ("omitted","Письменный эпизод не пересказан."),
 ("uncertain","Смысл ответа или отдельные слова остаются неоднозначными.")]], [175,320])
add("Далее приведены ключевые сопоставления. Полная таблица с цитатами и таймкодами доступна в приложении и CSV.")

for pid in order:
    story.append(PageBreak())
    heading(names.get(pid,pid))
    rows=[]
    for src in [s for s in sources if s["participant_id"]==pid]:
        related=[n for n in notes if n["story_id"]==src["story_id"]]
        cats=list(dict.fromkeys(relation.get(x["relation"],x["relation"]) for x in related))
        rows.append([story_topic_title(src["title"]),"; ".join(cats) or "Не найден соответствующий эпизод"])
    if rows:table(f"Результаты сверки историй: {names.get(pid,pid)}",["Письменная история","Результат сверки"],rows,[240,255])
    rank={"contradiction_candidate":0,"uncertain":1,"different":2,"consistent":3,"omitted":4}
    chosen=sorted([n for n in notes if n["participant_id"]==pid and n.get("spoken_excerpt")],key=lambda n:(rank.get(n["relation"],9),-float(n.get("confidence",0))))[:3]
    evidence_groups={}
    for n in chosen:
        trial=trial_by_comparison.get(n["comparison_id"],{})
        evidence_key=trial.get("trial_id") or n.get("qa_id") or (n.get("start_s"),n.get("end_s"),n.get("spoken_excerpt"))
        evidence_groups.setdefault(evidence_key,[]).append(n)
    for group in evidence_groups.values():
        n=group[0]
        trial=trial_by_comparison.get(n["comparison_id"],{})
        start,end=trial.get("start_s",n.get("start_s")),trial.get("end_s",n.get("end_s"))
        add(f"{relation.get(n['relation'],n['relation'])} · {timecode(start)}–{timecode(end)}","Sub")
        quotes=list({(story_topic_title(e["story_title"]),trim(e.get("source_excerpt"),230)):e for e in group}.values())
        for evidence in quotes:
            label=f"Эталон «{story_topic_title(evidence['story_title'])}»" if len(quotes)>1 else "Эталон"
            add(f"{label}: «{trim(evidence.get('source_excerpt'),230)}»","SmallRu")
        if n.get("question_summary"):
            add(f"Смысл вопроса: {trim(n['question_summary'],230)}","SmallRu")
        elif n.get("question_context"):
            add(f"Вопрос: «{trim(n['question_context'],230)}»","SmallRu")
        add(f"Ответ: «{trim(n.get('spoken_excerpt'),230)}»","SmallRu")
        for extra in n.get("additional_spoken_evidence",[]):
            question=(f" {extra['question_context']}" if extra.get("question_context") else "")
            add(f"Следующая реплика · {timecode(extra.get('start_s'))}:{question} «{trim(extra.get('spoken_excerpt'),160)}»","SmallRu")
        if n["relation"] in {"uncertain", "different"}:
            add(trim(n.get("interpretation"),370),"SmallRu")

story.append(PageBreak())
heading("Статистический анализ")
if not multimodal.get("models"):
    raise SystemExit("Run scripts/analyze_multimodal.py before building the report")
dataset=multimodal["dataset"]
model_rows=multimodal["models"]
model_metrics={row["id"]:row["metrics"] for row in model_rows}
increment=multimodal["profile_increment"]
fusion_increment=multimodal["fusion_profile_increment"]
trial_features=csvrows(OUT/"research/multimodal/trial_features.csv")
feature_counts={key:sum(num(row.get(key))!="н/д" for row in trial_features) for key in [
    "audio_pitch_median_hz", "video_head_speed_norm_per_s", "video_mouth_motion_ratio_per_s",
    "bio_ecg_hr_bpm_context120s", "bio_rip_rate_bpm_context120s"]}
add(f"Для проверки моделей отобраны {dataset['trials']} полных ответов с метками по письменному эталону: соответствий — {dataset['agreements']}, несоответствий — {dataset['contradictions']}. Оба класса представлены у всех {dataset['participants']} участников. Единица анализа — один ответ; при нескольких проверяемых утверждениях достаточно одного несоответствия для метки 1.")
add("Проверка LOPO каждый раз исключает из обучения целиком одного участника [3]. Заполнение пропусков и масштабирование обучаются только на остальных. L2-логистическая регрессия: C = 0,1, порог 0,5, без подбора параметров. Участники имеют равный вес при обучении и расчёте метрик.","SmallRu")
add("Речевые признаки: основной тон, его разброс, уровень сигнала, темп и доля пауз. Видеопризнаки: скорость головы, движение торса, раскрытие глаз и движение рта. БиоРадио: ЧСС, RMSSD ЭКГ и частота RIP в 120-секундном контексте ответа.","SmallRu")
add(f"До заполнения пропусков основной тон доступен для {feature_counts['audio_pitch_median_hz']} ответов, движение головы — для {feature_counts['video_head_speed_norm_per_s']}, рта — для {feature_counts['video_mouth_motion_ratio_per_s']}, ЧСС ЭКГ — для {feature_counts['bio_ecg_hr_bpm_context120s']}, частота RIP — для {feature_counts['bio_rip_rate_bpm_context120s']}.","SmallRu")
add("Профили включают выраженность, Эмоц+, Эмоц− и отдельные признаки наличия компонентов естественного и демонстрируемого типа. Эмоциональный баланс используется в корреляциях. Имена, ИСУ и тексты вопросов и эталонов в признаки моделей не входят.","SmallRu")
table("Качество моделей при проверке LOPO",["Модель","Brier","95 % ДИ Brier","Сбаланс. точность","ROC AUC"],[
 [model_name(row["name"]),num(row["metrics"].get("brier"),3),
  f"[{num(row['metrics'].get('brier_ci_low'),3)}; {num(row['metrics'].get('brier_ci_high'),3)}]",
  num(row["metrics"].get("balanced_accuracy"),3),num(row["metrics"].get("roc_auc"),3)]
 for row in model_rows],[185,50,100,85,75])
add(f"Brier — средняя квадратичная ошибка вероятности; меньше лучше. Сбалансированная точность — среднее чувствительности и специфичности; ROC AUC — качество ранжирования ответов. Для последних двух метрик случайный уровень равен 0,5. 95 % ДИ рассчитаны bootstrap по участникам ({increment.get('bootstrap',5000)} повторов) при фиксированных прогнозах LOPO, без повторного обучения.","SmallRu")
add(f"Среди проверенных наборов наименьшая ошибка у видеопризнаков: Brier {num(model_metrics['video']['brier'],3)} против {num(model_metrics['prior']['brier'],3)} у прогноза по доле классов. Однако ROC AUC = {num(model_metrics['video']['roc_auc'],3)}, а сбалансированная точность = {percent(model_metrics['video']['balanced_accuracy'])}: разделение ответов осталось около случайного уровня.","SmallRu")

story.append(PageBreak())
heading("Вклад профилей и биосигналов")
fig=OUT/"research/multimodal/model_comparison.png"
if not fig.exists():raise SystemExit(f"Missing analysis figure: {fig}")
figure(fig,"Ошибка на исключённых участниках и изменение при добавлении профилей. Ромбы показывают средние; отдельные точки отражают участников")
add(f"Добавление профилей к видео и речи изменило Brier на {num(increment['delta_brier'],4)}; 95 % ДИ [{num(increment['ci_low'],4)}; {num(increment['ci_high'],4)}]. Положительная разность означает рост ошибки. Интервал включает ноль; преимущество профилей не установлено.","SmallRu")
classified=model_metrics['av_profiles']['confusion_counts']
add(f"Модель «видео + речь + профили» обнаружила {classified['tp']} из {dataset['contradictions']} {plural(dataset['contradictions'],'несоответствия','несоответствий','несоответствий')}, пропустила {classified['fn']} и дала {counted(classified['fp'],'ложное срабатывание','ложных срабатывания','ложных срабатываний')}. Добавление БиоРадио снизило Brier с {num(model_metrics['av']['brier'],3)} до {num(model_metrics['fusion']['brier'],3)}, но ROC AUC осталось {num(model_metrics['fusion']['roc_auc'],3)}. Объединение каналов не обеспечило распознавания.","SmallRu")
add("Проверена чувствительность результатов к временному совмещению: расчёт повторён на нижней и верхней границах интервалов автоматического смещения при scale = 1.","SmallRu")
table("Чувствительность моделей к границам смещения",["Граница смещения","Модель","Brier","ROC AUC"],[
 [scenario["name"],model_name(row["name"]),
  num(row["metrics"].get("brier"),3),num(row["metrics"].get("roc_auc"),3)]
 for scenario in multimodal["sync_sensitivity"] for row in scenario["models"]],[145,170,90,90])
add(f"При обоих смещениях ROC AUC остаётся ниже 0,5. Для полного набора каналов добавление профилей даёт Δ Brier = {num(fusion_increment['delta_brier'],6)}, 95 % ДИ [{num(fusion_increment['ci_low'],4)}; {num(fusion_increment['ci_high'],4)}]; улучшение также не установлено.","SmallRu")

story.append(PageBreak())
heading("Связи профилей с измерениями")
families=multimodal["association_families"]
associations=multimodal["associations"]
estimated=sum(row.get("rho") is not None for row in associations)
add(f"Сопоставлены профили и показатели за всю запись. Единица анализа — участник, n = {dataset['participants']}. Заданы {families['primary']} основных сочетаний (выраженность и эмоциональный баланс с 6 признаками) и {families['secondary']} дополнительных. Коэффициент Спирмена, точные двусторонние перестановки рангов, поправка Холма отдельно в каждом семействе; 95 % ДИ — парный bootstrap участников [4].","SmallRu")
add(f"Рассчитаны {estimated} из {len(associations)} сочетаний; остальные не набрали минимум 6 пригодных участников или вариативных значений. Для ЭКГ, RIP и rPPG требуется не менее 20 % пригодных окон записи. После поправки Холма статистически значимых связей не обнаружено (p < 0,05).","SmallRu")
fig=OUT/"research/multimodal/associations.png"
if not fig.exists():raise SystemExit(f"Missing analysis figure: {fig}")
figure(fig,f"Все {families['primary']} основных сочетаний. Одна точка соответствует одному участнику; подписи содержат n, коэффициент Спирмена и p Холма",max_height=540)
head_association=next(row for row in associations if row['profile']=='expression' and row['feature']=='video_head_speed_norm_per_s')
add(f"Самая сильная по модулю связь в основном семействе — выраженность профиля и скорость головы: ρ = {num(head_association['rho'],3)}, p Холма = {num(head_association['p_holm'],3)}. Более высокой выраженности соответствовала меньшая скорость движения головы; связь не достигла статистической значимости. Здесь сравниваются различия между участниками за всю запись; связь этого признака с меткой ответа отдельно не установлена.","SmallRu")

story.append(PageBreak())
heading("Интерфейс и запуск")
add("Локальное приложение объединяет пять зон: временную шкалу; стенограмму и истории; биосигналы; видео с областями лица и позой; спектр ВСР и численные признаки. Переход по цитате открывает нужный фрагмент с вопросом. Разметка и смещение сохраняются в JSON.")
figure(OUT/"report/interface.png","Интерфейс просмотра и сопоставления записей",max_height=340)
add("Запуск: start.cmd. Установка для Windows/Linux/macOS и запуск через Docker описаны в README.")
story.append(Paragraph("Исходный код, тесты и конфигурация GitHub Actions: "+link(args.repository)+".",styles["TextRu"]))
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
table("Файлы результатов",["Путь в проекте","Содержимое"],[
 ["outputs/research/content_comparison.csv","Сопоставления с цитатами, контекстом и временем"],
 ["outputs/research/multimodal/analysis-results.zip","Полный анализ: CSV, графики, план, происхождение, параметры и код"],
 ["outputs/research/multimodal/associations.csv","Все корреляции: n, rho, p, поправка Холма и интервалы"],
 ["outputs/research/multimodal/participant_features.csv","Показатели участников за всю запись"],
 ["outputs/research/multimodal/trial_features.csv","Показатели отдельных ответов"],
 ["outputs/research/multimodal/model_predictions.csv","Прогнозы LOPO для воспроизведения оценок"],
 ["outputs/audio/","Стенограммы, временные метки, акустические признаки"],
 ["outputs/bio/","ЭКГ, ФПГ, RIP, GSR, интервалы и показатели ВСР"],
 ["outputs/video/","Лицевые области, rPPG и кинематика"],
 ["outputs/sync/","Фоновые участки и оценки смещения"],
 ["data/private/","Письменные эталоны и каталог источников"],
 ["docs/validation.md","Порядок технических проверок"],
],[235,260])

story.append(PageBreak())
heading("Выводы по работе")
add("Создан комплекс обработки и синхронного просмотра видео, речи и БиоРадио. Рассчитаны характеристики голоса, движений, пульса, дыхания и вариабельности сердечного ритма. Сопоставление устных ответов с письменными историями выявило несоответствия у каждого участника; для них сохранены проверяемые цитаты и временные границы.")
add("Автоматическое распознавание лжи по измеренным признакам на этих данных не достигнуто. Лучший из проверенных наборов — движения головы и торса, раскрытие глаз и движение рта — дал качество около случайного уровня. Речевые признаки, биосигналы и их объединение не обеспечили улучшения относительно простого прогноза по доле классов. Отдельный надёжный маркер лжи среди рассчитанных показателей не установлен.")
add("Добавление профилей участников не снизило ошибку вероятностного прогноза Brier. Корреляционный анализ профилей с показателями за всю запись также не выявил значимых связей после поправки на множественные проверки.")
add("Сравнение контактных каналов показало близкие оценки ЧСС по ФПГ и ЭКГ в прошедших контроль окнах. Точность видеопульса заметно различалась между записями. Для классификации основную полноту данных обеспечивала кинематика; акустические и биосигнальные признаки содержали пропуски.")

def footer(c,doc):
    c.setFont("Body",8);c.setFillColor(colors.HexColor(MUTED))
    c.drawRightString(A4[0]-50,28,str(doc.page))

path=args.output
path.parent.mkdir(parents=True,exist_ok=True)
doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=42,bottomMargin=45,
                     title="Лабораторная работа НТИ. Анализ видео, речи и биосигналов",
                     author=", ".join(name for name,_ in authors))
doc.build(story,onFirstPage=lambda c,doc:None,onLaterPages=footer)
print(path)
