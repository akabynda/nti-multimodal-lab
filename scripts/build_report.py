"""Build the local Russian laboratory report from calculated outputs.

Uses reportlab and an available Cyrillic TrueType font; private results are never
embedded into the repository. Run after processing and validation.
"""
from pathlib import Path
import argparse
import json
import csv
import math
import pandas as pd
from collections import Counter
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, PageBreak, Image, KeepTogether)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from nti_lab.palette import BLUE, INK, MUTED, LINE, SURFACE, WHITE
from nti_lab.story_topics import story_topic_title
from nti_lab.multimodal_analysis import TYPE_CORRECTIONS, FEATURE_LABELS
from nti_lab import contrasts as contrast_lib
from nti_lab import session_summary as summary_lib
from nti_lab import report_figures
import numpy as np

TITLE="Мультимодальное распознавание лжи на основе синхронного анализа видеоряда, аудиопотока и БОС"
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
styles.add(ParagraphStyle("ListRu",fontName="Body",fontSize=10,leading=14,spaceAfter=2,leftIndent=14,firstLineIndent=-10))
styles.add(ParagraphStyle("TocTitle",parent=styles["Section"],alignment=TA_CENTER))
styles.add(ParagraphStyle("Toc0",fontName="Body",fontSize=10,leading=15,leftIndent=0))
styles.add(ParagraphStyle("Toc1",fontName="Body",fontSize=9.5,leading=13,leftIndent=18))
story=[]
numbers={"table":0,"figure":0}
def p(text,style="TextRu"):
    return Paragraph(escape(str(text)).replace("\n","<br/>"),styles[style])
def add(text,style="TextRu"):
    story.append(p(text,style))
def contents_entry(text,style,level):
    entry=p(text,style);entry.toc_level=level
    story.append(entry)
SECTION_NUMBERS={}
def set_outline(outline):
    """Number chapters and subsections from the declared outline; titles are unique."""
    for i,(chapter,subsections) in enumerate(outline,1):
        SECTION_NUMBERS[chapter]=str(i)
        for j,title in enumerate(subsections,1):
            SECTION_NUMBERS[title]=f"{i}.{j}"
def ref(title):
    return SECTION_NUMBERS[title]
def heading(text,numbered=True):
    contents_entry(f"{ref(text)} {text}" if numbered else text,"Section",0)
def subheading(text):
    contents_entry(f"{ref(text)} {text}","Sub",1)
def next_figure():
    return numbers["figure"]+1
REFERENCES={
 "hrv":"Heart rate variability: standards of measurement, physiological interpretation and clinical use / Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology // Circulation. – 1996. – Vol. 93, № 5. – P. 1043–1065. – DOI 10.1161/01.CIR.93.5.1043.",
 "qrs":"Pan J., Tompkins W. J. A real-time QRS detection algorithm // IEEE Transactions on Biomedical Engineering. – 1985. – Vol. BME-32, № 3. – P. 230–236. – DOI 10.1109/TBME.1985.325532.",
 "welch":"Welch P. D. The use of fast Fourier transform for the estimation of power spectra: a method based on time averaging over short, modified periodograms // IEEE Transactions on Audio and Electroacoustics. – 1967. – Vol. 15, № 2. – P. 70–73. – DOI 10.1109/TAU.1967.1161901.",
 "mediapipe":"MediaPipe Face Landmarker, Pose Landmarker : документация / Google. – URL: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker.",
 "pos":"Wang W., den Brinker A. C., Stuijk S., de Haan G. Algorithmic principles of remote PPG // IEEE Transactions on Biomedical Engineering. – 2017. – Vol. 64, № 7. – P. 1479–1491. – DOI 10.1109/TBME.2016.2609282.",
 "whisper":"Radford A., Kim J. W., Xu T. [et al.] Robust speech recognition via large-scale weak supervision // Proceedings of the 40th International Conference on Machine Learning. – 2023. – P. 28492–28518.",
 "faster_whisper":"faster-whisper : программная библиотека / SYSTRAN. – URL: https://github.com/SYSTRAN/faster-whisper.",
 "silero":"Silero VAD : pre-trained voice activity detector / Silero Team. – URL: https://github.com/snakers4/silero-vad.",
 "parselmouth":"Jadoul Y., Thompson B., de Boer B. Introducing Parselmouth: a Python interface to Praat // Journal of Phonetics. – 2018. – Vol. 71. – P. 1–15. – DOI 10.1016/j.wocn.2018.07.001.",
 "wespeaker":"Wang H., Liang C., Wang S. [et al.] WeSpeaker: a research and production oriented speaker embedding learning toolkit // ICASSP 2023 – IEEE International Conference on Acoustics, Speech and Signal Processing. – 2023. – P. 1–5. – DOI 10.1109/ICASSP49357.2023.10096626.",
 "baevsky":"Баевский Р. М., Иванов Г. Г., Чирейкин Л. В. [и др.] Анализ вариабельности сердечного ритма при использовании различных электрокардиографических систем (методические рекомендации) // Вестник аритмологии. – 2001. – № 24. – С. 65–87.",
 "depaulo":"DePaulo B. M., Lindsay J. J., Malone B. E. [et al.] Cues to deception // Psychological Bulletin. – 2003. – Vol. 129, № 1. – P. 74–118. – DOI 10.1037/0033-2909.129.1.74.",
 "vrij":"Vrij A. Detecting Lies and Deceit: Pitfalls and Opportunities. – 2nd ed. – Chichester : John Wiley & Sons, 2008. – 488 p.",
 "verkruysse":"Verkruysse W., Svaasand L. O., Nelson J. S. Remote plethysmographic imaging using ambient light // Optics Express. – 2008. – Vol. 16, № 26. – P. 21434–21445. – DOI 10.1364/OE.16.021434.",
 "chrom":"de Haan G., Jeanne V. Robust pulse rate from chrominance-based rPPG // IEEE Transactions on Biomedical Engineering. – 2013. – Vol. 60, № 10. – P. 2878–2886. – DOI 10.1109/TBME.2013.2266196.",
 "praat":"Boersma P., Weenink D. Praat: doing phonetics by computer : программа. – URL: https://www.praat.org.",
 "blazepose":"Bazarevsky V., Grishchenko I., Raveendran K. [et al.] BlazePose: on-device real-time body pose tracking // arXiv. – 2020. – arXiv:2006.10204.",
 "merrill":"Merrill D. W., Reid R. H. Personal Styles and Effective Performance. – Radnor : Chilton Book Company, 1981. – 252 p.",
 "lopo":"Cross-validation: evaluating estimator performance. LeaveOneGroupOut : документация scikit-learn. – URL: https://scikit-learn.org/stable/modules/cross_validation.html.",
 "brier":"Brier G. W. Verification of forecasts expressed in terms of probability // Monthly Weather Review. – 1950. – Vol. 78, № 1. – P. 1–3. – DOI 10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2.",
 "holm":"Holm S. A simple sequentially rejective multiple test procedure // Scandinavian Journal of Statistics. – 1979. – Vol. 6, № 2. – P. 65–70.",
 "permutation":"scipy.stats.permutation_test : документация SciPy. – URL: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html.",
}
cited=[]
def cite(*keys):
    for key in keys:
        if key not in cited:cited.append(key)
    return "["+", ".join(str(cited.index(key)+1) for key in keys)+"]"
def link(url):
    return f'<a href="{escape(url)}" color="{BLUE}">{escape(url)}</a>'
# Lead-in sentences referencing a table; {n} is the table number.
TABLE_INTROS={
 "Сокращения":"Сокращения, используемые в отчёте, приведены в таблице {n}.",
 "Регистрируемые данные":"Состав регистрируемых данных приведён в таблице {n}.",
 "Профили участников":"Поведенческие профили участников приведены в таблице {n}, эмоциональные показатели — в таблице {m}.",
 "Этапы сессии, с от начала видео":"Границы этапов приведены в таблице {n}, порядок тематических блоков — в таблице {m}.",
 "Окна ВСР и контроль качества ЭКГ":"Число окон ВСР и окон, прошедших контроль качества ЭКГ, приведено в таблице {n}.",
 "Показатели ВСР по ЭКГ: медианы окон, прошедших контроль":"Временные и спектральные показатели ВСР приведены в таблице {n}, частота дыхания по RIP — в таблице {m}.",
 "Сравнение контактного ФПГ и ЭКГ":"Результаты сравнения приведены в таблице {n}.",
 "Покрытие лица, видеопульс и дыхательное движение":"Покрытие лица и согласие видеопульса и дыхательного движения с ЭКГ и RIP приведены в таблице {n}.",
 "Поза и корпус":"Показатели позы и корпуса приведены в таблице {n}, головы и кистей — в таблице {m}.",
 "Редакторская проверка стенограмм":"Объём редакторской проверки приведён в таблице {n}.",
 "Временные параметры диалога":"Временные параметры диалога приведены в таблице {n}, акустические характеристики — в таблице {m}.",
 "Категории сверки с письменным эталоном":"Категории сверки приведены в таблице {n}, их число — в таблице {m}.",
 "Число ответов по меткам":"Число ответов с каждой меткой приведено в таблице {n}.",
 "Проверяемые гипотезы, уровень значимости 0,05":"Гипотезы, методы проверки и результаты приведены в таблице {n}.",
 "Модели LOPO на исходных и нормированных к «Базе» признаках":"Качество моделей на исходных и нормированных признаках приведено в таблице {n}.",
 "Качество моделей при проверке LOPO":"Качество моделей приведено в таблице {n}.",
 "Чувствительность моделей к границам смещения":"Результаты приведены в таблице {n}.",
 "Зоны интерфейса":"Содержимое зон приведено в таблице {n}.",
 "Модули обработки":"Модули программного комплекса, их назначение и использованные модели приведены в таблице {n}.",
 "Порядок обработки":"Команды шагов обработки приведены в таблице {n}; параметры — в README и docs/.",
 "Файлы результатов":"Файлы результатов приведены в таблице {n}.",
}
def table(caption,headers,rows,widths):
    if caption in TABLE_INTROS:
        add(TABLE_INTROS[caption].format(n=numbers["table"]+1,m=numbers["table"]+2))
        story[-1].keepWithNext=True
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
def figure(path,caption,max_height=None,intro=None):
    if not path.exists():return
    im=Image(str(path));scale=495/im.imageWidth
    if max_height:scale=min(scale,max_height/im.imageHeight)
    im.drawWidth=im.imageWidth*scale;im.drawHeight=im.imageHeight*scale
    numbers["figure"]+=1
    lead=[p(intro.format(n=numbers["figure"]))] if intro else []
    # A heading directly before a figure moves to the figure's page with it.
    if story and getattr(story[-1],"toc_level",None) is not None:lead.insert(0,story.pop())
    story.append(KeepTogether(lead+[im,p(f"Рисунок {numbers['figure']} – {caption}","FigureCaption")]))
    return numbers["figure"]
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
    # Adding 0.0 turns a rounded negative zero into zero.
    try:return decimal(f"{round(float(v),d)+0.0:.{d}f}") if math.isfinite(float(v)) else "н/д"
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
set_outline([
 ("Описание эксперимента",["Участники и протокол сессии","Регистрируемые данные","Записи участников","Профили участников","Этапы сессии"]),
 ("Аналитический обзор",["Признаки лжи","Дистанционная фотоплетизмография","Вариабельность сердечного ритма",
                         "Речь и диалог","Поза и движения","Проверка на малой выборке"]),
 ("Архитектура программного комплекса",["Схема обработки","Модули и библиотеки","Репозиторий и запуск"]),
 ("Синхронизация",[]),
 ("Биосигналы БиоРадио",["ЭКГ, ВСР и дыхание"]+(["Контактный ФПГ и ЭКГ"] if ppg_records else [])),
 ("Видеоаналитика",["Лицевые области, видеопульс и дыхание","Кинематика тела, головы и конечностей"]),
 ("Речевой анализ",["Распознавание речи и проверка стенограмм","Речевые и акустические показатели"]),
 ("Разметка ответов",["Сопоставление с письменным эталоном"]),
 ("Статистический анализ",["Метки ответов и нормализация","Гипотезы","Контрасты и маркеры по группам профиля",
                           "Модели классификации","Вклад профилей и биосигналов","Связи профилей с измерениями"]),
 ("Интерфейс",[])])
FIGURES=OUT/"report/figures"
summaries={}
for a in audios:
    rid,pid=a["recording_id"],a["participant_id"]
    video=by_video.get(pid,{})
    row={"recording_id":rid,"participant_id":pid}
    if (OUT/"bio"/pid/"metrics.csv").exists():
        row.update(summary_lib.hrv_summary(pd.read_csv(OUT/"bio"/pid/"metrics.csv")))
    if (OUT/"audio"/f"{rid}.features.csv").exists():
        row.update(summary_lib.speech_summary(a,pd.read_csv(OUT/"audio"/f"{rid}.features.csv")))
    if (OUT/"video"/f"{rid}.kinematics.csv").exists():
        row.update(summary_lib.kinematics_summary(pd.read_csv(OUT/"video"/f"{rid}.kinematics.csv"),video.get("overlays",[])))
    stages=summary_lib.session_stages(a,by_alignment.get(pid,{}).get("baselines",[]))
    row.update(stages=stages,neutral_answers=len(stages["neutral_answers"]),topic_blocks=len(stages["blocks"]))
    summaries[pid]=row
(OUT/"report").mkdir(parents=True,exist_ok=True)
pd.DataFrame([{k:v for k,v in row.items() if k!="stages"} for row in summaries.values()]).to_csv(
    OUT/"report/session_summary.csv",index=False,encoding="utf-8-sig")
def summary(pid,key,d=1,scale=1):
    value=summaries.get(pid,{}).get(key)
    return num(value*scale if value is not None else None,d)

def profile_types(raw):
    tokens=[TYPE_CORRECTIONS.get(t.strip().lower(),t.strip().lower()) for t in str(raw or "").split("/") if t.strip()]
    return "/\n".join(t.capitalize() for t in tokens) or "н/д"
def surname_first(name):
    given,_,family=name.partition(" ")
    return f"{family} {given}" if family else name
authors=sorted([surname_first(private_names[pid]),by_profile.get(pid,{}).get("tab_number") or "н/д"]
               for pid in order if pid in private_names)
add("Университет ИТМО","CoverOrg")
story.append(Spacer(1,170))
add("Отчёт по лабораторной работе","CoverSubtitle")
story.append(Spacer(1,10))
add(TITLE,"ReportTitle")
story.append(Spacer(1,90))
if authors:
    cover=Table([[p("Выполнили:","Sub"),""]]+[[p(name),p(f"ИСУ {isu}")] for name,isu in authors],
                colWidths=[170,90],hAlign="RIGHT")
    cover.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),0),("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    story.append(cover)
story.append(Spacer(1,140))
story.append(Paragraph("Репозиторий проекта: "+link(args.repository),styles["CoverText"]))
story.append(PageBreak())
add("Содержание","TocTitle")
contents=TableOfContents(dotsMinLevel=0);contents.levelStyles=[styles["Toc0"],styles["Toc1"]]
story.append(contents)
story.append(PageBreak())
add("Перечень сокращений","TocTitle")
abbreviations=[
 ("БОС","биологическая обратная связь; запись прибора БиоРадио"),("ВСР","вариабельность сердечного ритма"),
 ("ЧСС","частота сердечных сокращений"),("ЧДД","частота дыхательных движений"),
 ("ЭКГ","электрокардиограмма"),("ФПГ","контактная фотоплетизмограмма"),
 ("rPPG","дистанционная фотоплетизмограмма по видео лица"),("POS","метод rPPG Plane-Orthogonal-to-Skin"),
 ("ROI","область интереса на лице"),("RIP","дыхательный пояс (респираторная индуктивная плетизмография)"),
 ("GSR","кожно-гальваническая реакция"),("EDR","дыхание, рассчитанное по ЭКГ"),
 ("RR","интервал между соседними R-зубцами ЭКГ"),("SDNN","стандартное отклонение RR-интервалов"),
 ("RMSSD","корень из среднего квадрата разностей соседних RR"),("pNN50","доля разностей соседних RR больше 50 мс"),
 ("ULF, VLF, LF, HF","ультранизко-, очень низко-, низко- и высокочастотные диапазоны спектра ВСР"),
 ("F0","основной тон голоса"),("VAD","обнаружение речевой активности"),
 ("ASR","автоматическое распознавание речи"),("LOPO","проверка с исключением одного участника"),
 ("MAE","средняя абсолютная ошибка"),("ROC AUC","площадь под ROC-кривой"),
 ("ДИ","доверительный интервал"),("ИН","индекс напряжения регуляторных систем Баевского"),
 ("ИСУ","номер в информационной системе университета")]
table("Сокращения",["Сокращение","Расшифровка"],[list(x) for x in abbreviations],[110,385])
story.append(PageBreak())
heading("Введение",numbered=False)
add("Контактная полиграфия регистрирует дыхание, пульс и кожно-гальваническую реакцию датчиками на теле. Камера и микрофон дают бесконтактные каналы: пульс по изменению цвета кожи лица, движения тела и головы, временные и акустические параметры речи. Сведение контактных и бесконтактных каналов на одну шкалу времени позволяет сравнить реакции разных систем на одном и том же ответе.")
add("Реакции на ложь и стресс различаются между людьми. Поэтому признаки ответов сравниваются с индивидуальной нормой участника, а модели проверяются на участниках, не входивших в обучение.")
add("Цель работы: разработать программный комплекс для контактного и бесконтактного выявления признаков психоэмоционального напряжения и скрываемой информации по синхронизированным видео, аудио и биосигналам.")
add("Технические задачи:","Sub")
for item in ["синхронизировать видео, аудио, ЭКГ, ФПГ и дыхательные сигналы;",
             "рассчитать видеопульс rPPG и частоту дыхания по видео лица и плечевого пояса;",
             "рассчитать латентность и длительность ответа, темп речи, основной тон, энергию и спектральные характеристики голоса;",
             "отследить движения головы, корпуса, рук и ног; продолжать анализ при отсутствии части тела в кадре;",
             "рассчитать ВСР по ЭКГ и ФПГ в окне 120 с с шагом 5 с и сравнить контактные и бесконтактные оценки;",
             "разработать интерфейс из пяти синхронизированных зон;",
             "оформить открытый репозиторий с автоматической проверкой и воспроизводимым запуском."]:
    add("– "+item,"ListRu")
add("Исследовательские задачи:","Sub")
for item in ["разметить ответы участников по письменным правдивым историям: «Правда», «Ложь», «База»;",
             "сформулировать и проверить гипотезы о различиях реакций на ложные и правдивые ответы и о влиянии поведенческого профиля участника;",
             "выявить наиболее информативные маркеры для групп профилей;",
             "построить классификационную модель и оценить вклад профиля в качество распознавания."]:
    add("– "+item,"ListRu")
story.append(PageBreak())
heading("Описание эксперимента")
subheading("Участники и протокол сессии")
recording_days=Counter(b.get("recording_start_time","")[:10] for b in bio if b.get("recording_start_time"))
day_text="; ".join(f"{d[8:10]}.{d[5:7]}.{d[:4]} — {counted(n,'участник','участника','участников')}" for d,n in sorted(recording_days.items()))
add(f"В эксперименте участвовали {counted(len(order),'человек','человека','человек')} — авторы отчёта. Даты записи: {day_text}. Каждая сессия записана одной камерой и прибором БиоРадио.")
topic_counts=Counter(story_topic_title(x["title"]) for x in sources)
common=[t for t,n in topic_counts.items() if n==len(order)]
other=[t for t,n in topic_counts.most_common() if n<len(order)]
per_person=round(len(sources)/len(order)) if order else 0
add(f"Каждый участник написал {per_person} правдивых историй о себе. Темы {', '.join('«'+t+'»' for t in common)} есть у всех; остальные выбраны из тем {', '.join('«'+t+'»' for t in other)}. Истории служат эталоном для разметки ответов (раздел {ref('Разметка ответов')}).")
add("Сессия: начальный фон 2 мин с закрытыми глазами; интервью с нейтральными вопросами и вопросами по темам историй; конечный фон 2 мин. В интервью участники намеренно изменяли часть сведений относительно своих историй. Интервалы ложных ответов участниками не размечались.")
subheading("Регистрируемые данные")
channel_names={"ECG":"ЭКГ","RIP":"RIP","GSR":"GSR","PPG PulseOx1":"контактный ФПГ","SpO2 PulseOx1":"SpO2",
               "Heart Rate PulseOx1":"ЧСС пульсоксиметра","Flow Rate Sensor 1":"датчик потока воздуха"}
bio_groups=Counter()
for pid in order:
    meta=read(OUT/"bio"/pid/"metadata.json",{})
    channels=meta.get("channels",[])
    if channels:
        rates=sorted({c["sample_rate_hz"] for c in channels})
        bio_groups[(", ".join(channel_names.get(c["name"],c["name"]) for c in channels),"/".join(plain(x) for x in rates))]+=1
video=by_video.get(order[0],{}) if order else {}
data_rows=[["Видео",f"{video.get('source_file','').rsplit('.',1)[-1].upper()}, {video.get('source_width')}×{video.get('source_height')}, {plain(video.get('source_fps'))} кадр/с; анализ лица {num(video.get('analysis_fps'),1)} кадр/с, позы до 5 Гц",str(len(videos))],
           ["Звук","Дорожка видеофайла; для ASR и акустики декодируется с частотой 16 кГц",str(len(audios))]]
data_rows+=[[f"БиоРадио",f"{names_} ; {rate} Гц".replace(" ;",";"),str(n)] for (names_,rate),n in bio_groups.items()]
data_rows+=[["Письменные истории",f"{counted(len(sources),'история','истории','историй')} из {counted(research.get('source_documents',0),'исходного документа','исходных документов','исходных документов')}",str(len({x['participant_id'] for x in sources}))],
            ["Анкета","Результаты.xlsx: тест Л34, профиль поведения",str(sum(pid in by_profile for pid in order))]]
table("Регистрируемые данные",["Поток","Параметры","Участников"],data_rows,[110,305,80])
add(f"После обработки получено {counted(sum(len(a.get('qa_candidates',[])) for a in audios),'пара','пары','пар')} «вопрос – ответ» и {counted(total_windows,'окно','окна','окон')} ВСР по 120 с с шагом 5 с.")
subheading("Записи участников")
rows=[]
for i,a in enumerate(audios,1):
    b=by_bio.get(a["participant_id"],{})
    duration=by_alignment.get(a["participant_id"],{}).get("media_duration_s",a.get("duration_s"))
    difference=b.get("analysis_duration_s",0)-duration if duration is not None and b.get("analysis_duration_s") is not None else None
    rows.append([f"P{i:02}",names[a["participant_id"]],num(duration),num(b.get("analysis_duration_s")),num(difference)])
recordings_table=table("Длительность видеозаписей и записей БиоРадио",["ID","Участник","Видео, с","БОС, с","Разница, с"],rows,[35,190,90,90,90])
add(f"Видео, тексты и записи прибора объединены по именам файлов и данным документов. Обозначения P01–P{len(order):02} в таблицах и на рисунках соответствуют порядку таблицы {recordings_table}.")
subheading("Профили участников")
add(f"Профиль получен тестом Л34 (Результаты.xlsx). Естественный и демонстрируемый стили — четыре социальных стиля модели Д. Меррилла и Р. Рида {cite('merrill')}: «Контролирующий», «Стимулирующий», «Содействующий», «Координирующий»; допускается сочетание двух стилей, первым указан ведущий. Числовые шкалы: выраженность стиля, Эмоц+, Эмоц− и Эмоц_Баланс = Эмоц+ − Эмоц−.")
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
subheading("Этапы сессии")
add(f"Сессия состоит из начального фона с закрытыми глазами (2 мин), диалога и конечного фона. Границы фонов найдены по речевой инструкции интервьюера и раскрытию глаз (раздел {ref('Синхронизация')}). Диалог — пары «вопрос – ответ» между концом начального и началом конечного фона.")
add(f"Тематический блок задаётся по двум последним предложениям вопроса: первое совпадение с ключевыми основами темы истории (профессия, экзамен, достижения, алкоголь, обман, конфликт, летний день, нерешённая проблема) или нейтральных вопросов (о себе, увлечения, самочувствие). Вопрос без совпадения продолжает текущий блок. Ответы нейтрального блока используются как «База» (раздел {ref('Метки ответов и нормализация')}).")
def stage_span(stages,name):
    item=next((b for b in stages["baselines"] if b["name"]==name),None)
    return f"{num(item['start_s'],0)}–{num(item['end_s'],0)}" if item else "н/д"
stages_table=table("Этапы сессии, с от начала видео",["Участник","Начальный фон","Диалог","Конечный фон","Блоков","Ответов «База»"],[
 [f"P{i+1:02}",stage_span(summaries[pid]["stages"],"beginning"),
  f"{num(summaries[pid]['stages']['dialogue_start_s'],0)}–{num(summaries[pid]['stages']['dialogue_end_s'],0)}",
  stage_span(summaries[pid]["stages"],"ending"),str(summaries[pid]["topic_blocks"]),str(summaries[pid]["neutral_answers"])]
 for i,pid in enumerate(order)],[70,95,95,95,60,80])
table("Последовательность тематических блоков",["Участник","Блоки по порядку"],[
 [f"P{i+1:02}"," → ".join(b["topic"] or "Без темы" for b in summaries[pid]["stages"]["blocks"])]
 for i,pid in enumerate(order)],[70,425])
story.append(PageBreak())
heading("Аналитический обзор")
subheading("Признаки лжи")
add(f"Обобщение исследований поведенческих признаков обмана {cite('depaulo')} показало, что ни один признак не сопровождает ложь надёжно: большинство связей слабые и неустойчивы между выборками. Устойчивее признаки когнитивной нагрузки: построение ложного ответа требует больше ресурсов, что проявляется в увеличении времени до начала ответа и числа пауз и в замедлении речи {cite('vrij')}. ЧСС, дыхание и кожно-гальваническая реакция отражают общее возбуждение и не различают ложь и волнение; их изменения оцениваются относительно фона того же человека.")
subheading("Дистанционная фотоплетизмография")
add(f"Пульсовая волна меняет кровенаполнение кожи и вместе с ним её цвет. Изменение регистрируется обычной камерой; первые работы использовали зелёный канал {cite('verkruysse')}. Методы CHROM {cite('chrom')} и POS {cite('pos')} строят пульсовый сигнал из комбинации цветовых каналов, ортогональной изменениям яркости и бликам. Точность снижают движения головы, переменное освещение, сжатие видео и малая площадь кожи в кадре. В работе используется POS.")
subheading("Вариабельность сердечного ритма")
add(f"Стандарт анализа ВСР {cite('hrv')} задаёт временные показатели (SDNN, RMSSD, pNN50) и мощности спектра в диапазонах VLF, LF и HF. Индекс напряжения Р. М. Баевского {cite('baevsky')} рассчитывается по гистограмме RR-интервалов. По стандарту ULF оценивается только по суточным записям, а оценка VLF по записям 2–5 мин ненадёжна; в окне 120 с VLF приводится справочно.")
subheading("Речь и диалог")
add(f"Модель Whisper {cite('whisper')} распознаёт русскую речь и выдаёт временные метки слов. Разделение голосов строится по векторным представлениям диктора, например WeSpeaker {cite('wespeaker')}. Основной тон, форманты и интенсивность рассчитываются алгоритмами Praat {cite('praat')}. По временным меткам диалога рассчитываются латентность ответа, паузы и темп.")
subheading("Поза и движения")
add(f"Модели оценки позы по ключевым точкам, например BlazePose {cite('blazepose')}, определяют положение головы, плеч, кистей, таза и стоп в каждом кадре. По траекториям точек рассчитываются наклон корпуса, кивки и повороты головы, касания лица и закрытые позы. Точки с низкой уверенностью исключаются, остальные части тела анализируются независимо.")
subheading("Проверка на малой выборке")
add(f"Ответы одного человека зависимы, поэтому единица статистического вывода — участник. При {len(order)} участниках точные перестановочные тесты дают p-значения без предположения о нормальности. Модели проверяются с исключением всего участника из обучения {cite('lopo')}: иначе модель запоминает индивидуальные особенности. Поправка Холма {cite('holm')} ограничивает вероятность хотя бы одной ложной находки в семействе проверок.")
story.append(PageBreak())
heading("Архитектура программного комплекса")
subheading("Схема обработки")
report_figures.architecture(FIGURES/"architecture.png")
add("Комплекс обрабатывает каждую модальность отдельным модулем. Результаты модулей сохраняются в outputs/ и переводятся на единую шкалу времени видео; анализ и интерфейс читают сохранённые результаты. Повторный запуск шага пропускает записи с готовыми результатами, если не задан --force.")
figure(FIGURES/"architecture.png","Схема обработки данных программного комплекса",max_height=300,intro="Схема обработки показана на рисунке {n}.")
subheading("Модули и библиотеки")
table("Модули обработки",["Модуль","Назначение","Модели и библиотеки"],[
 ["biosignals.py","Чтение BCRX, R-зубцы, RR, ВСР, ИН, ФПГ, RIP, EDR, GSR","NumPy, SciPy"],
 ["audio.py, speakers.py, editorial.py","ASR, речевые интервалы, акустика, роли говорящих, пары «вопрос – ответ»","faster-whisper large-v3, Silero VAD, Parselmouth (Praat), WeSpeaker ResNet34 (sherpa-onnx)"],
 ["vision.py","Лицевые области, rPPG POS, кинематика тела","MediaPipe Face Landmarker и Pose Landmarker, OpenCV, PyAV"],
 ["alignment.py, sync.py","Поиск фонов, оценка смещения БиоРадио, преобразование времени","NumPy, SciPy"],
 ["research.py, story_topics.py","Каталог источников, сопоставление историй и стенограмм","scikit-learn (TF-IDF)"],
 ["session_summary.py","Сводные показатели записей и этапы сессии","pandas, SciPy"],
 ["multimodal_features.py, multimodal_analysis.py, contrasts.py","Признаки ответов, корреляции, модели LOPO, нормализация к «Базе», гипотезы","pandas, scikit-learn, SciPy"],
 ["app.py, static/","Интерфейс и API","FastAPI, Uvicorn, JavaScript"],
 ["report_figures.py, scripts/build_report.py","Рисунки и PDF отчёта","Matplotlib, ReportLab"]],[150,190,155])
subheading("Репозиторий и запуск")
story.append(Paragraph("Репозиторий: "+link(args.repository)+". Состав:",styles["TextRu"]))
for item in ["configs/pipeline.json — параметры выполненного расчёта;",
             "data/ — схема исходных данных и инструкция по размещению; сами данные в репозиторий не входят;",
             "src/nti_lab/ — модули обработки, анализа и интерфейса;",
             "scripts/ — шаги обработки и сборка отчёта;",
             "notebooks/research.ipynb — исследовательский анализ;",
             "tests/ — модульные тесты pytest;",
             "docs/ — методы, форматы и порядок проверок;",
             "environment.yml, requirements.txt, Dockerfile, docker-compose.yml — окружение Conda, зависимости pip и контейнер;",
             "README.md — системные требования, установка, порядок обработки и запуск интерфейса."]:
    add("– "+item,"ListRu")
table("Порядок обработки",["Шаг","Команда"],[
 ["БиоРадио","python scripts/analyze_bio.py"],
 ["Речь","python -m nti_lab.audio --input ../1_30 --output outputs/audio --skip-existing"],
 ["Видео","python scripts/analyze_video.py --input ../1_30 --workers 4"],
 ["Синхронизация","python scripts/estimate_sync.py"],
 ["Сверка историй","python scripts/analyze_research.py"],
 ["Признаки и модели","python scripts/analyze_multimodal.py --label-review … --assumption-helper …"],
 ["Нормализация и гипотезы","python scripts/analyze_contrasts.py"],
 ["Отчёт","python scripts/build_report.py"],
 ["Интерфейс","start.cmd или docker compose up --build -d"]],[130,365])
add("Тесты pytest и сборка Docker-образа выполняются GitHub Actions (.github/workflows/ci.yml) при каждом push.")
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
codes={pid:f"P{i+1:02}" for i,pid in enumerate(order)}
report_figures.sync_curves([(codes[pid],by_alignment[pid]) for pid in order if pid in by_alignment],FIGURES/"sync_curves.png")
figure(FIGURES/"sync_curves.png","Модуль коэффициента корреляции на начальном и конечном фоне и общая оценка в зависимости от опережения БиоРадио",max_height=520,
       intro="Кривые оценок по всему допустимому диапазону опережения показаны на рисунке {n}. Пунктир — принятое опережение (максимум общей оценки).")
supported=[pid for pid in order if by_alignment.get(pid,{}).get("status")=="supported_unverified"]
example=max(supported,key=lambda pid:by_alignment[pid]["joint_support"]["best_score"]) if supported else None
if example and summaries[example]["stages"]["neutral_answers"] and (OUT/"bio"/example/"signals.npz").exists():
    answer=summaries[example]["stages"]["neutral_answers"][0]
    window=(answer["answer_start_s"]-10,answer["answer_start_s"]+20)
    audio=next(a for a in audios if a["participant_id"]==example)
    report_figures.synchronized_signals(turns=audio.get("turns",[]),video=by_video[example],bio=dict(np.load(OUT/"bio"/example/"signals.npz")),
        lead_s=by_alignment[example]["joint_support"]["best_lead_s"],start_s=window[0],end_s=window[1],path=FIGURES/"synchronized_signals.png")
    figure(FIGURES/"synchronized_signals.png",f"Синхронные реплики, ЭКГ, rPPG и дыхание RIP, {codes[example]}, {num(window[0],0)}–{num(window[1],0)} с видео",max_height=440,
           intro=f"Пример сведения модальностей на одной оси времени видео ({codes[example]}, наибольшая общая оценка среди однозначных максимумов) показан на рисунке {{n}}. Биосигналы переведены по формуле tвидео = tБОС − опережение.")
story.append(PageBreak())
heading("Биосигналы БиоРадио")
subheading("ЭКГ, ВСР и дыхание")
add("Файлы BCRX декодированы в численные ряды. Частоты дискретизации, единицы и коэффициенты прочитаны из заголовков.")
add(f"R-зубцы найдены детектором по энергии комплекса QRS {cite('qrs')}: полосовой фильтр 5–25 Гц, скользящее среднее квадрата сигнала 60 мс, адаптивный порог 25 % от 95-го процентиля энергии в блоках 30 с, минимальный интервал между зубцами 300 мс. RR-интервал отклоняется, если он вне 300–2000 мс или отличается более чем на 20 % от медианы 11 соседних интервалов.")
add(f"По принятым RR рассчитаны ЧСС, SDNN, RMSSD, pNN50, VLF, LF, HF и LF/HF {cite('hrv')}. Спектр — метод Уэлча {cite('welch')} по тахограмме, интерполированной с частотой 4 Гц, окно Ханна. Окно ВСР — 120 с, шаг — 5 с. Окна с разрывами или нестабильными RR исключены из итоговых оценок. Диапазоны спектра: VLF 0,0033–0,04 Гц, LF 0,04–0,15 Гц, HF 0,15–0,4 Гц; общая мощность — сумма трёх диапазонов. ULF в окне 120 с не рассчитывается. Дыхательный пояс RIP обработан отдельно; EDR рассчитан по колебаниям амплитуды ЭКГ.")
table("Окна ВСР и контроль качества ЭКГ",["Участник","Окон","Прошли контроль","Медиана ЧСС, уд/мин"],[
 [f"P{i+1:02}",str(by_bio.get(pid,{}).get("window_count","—")),str(by_bio.get(pid,{}).get("usable_ecg_windows","—")),num(by_bio.get(pid,{}).get("median_hr_bpm"))]
 for i,pid in enumerate(order)],[80,110,145,160])
figure(OUT/"bio/aggregate_summary.png","Качество ЭКГ и описательные показатели по участникам",intro="Доля пригодных окон и распределения показателей ЭКГ приведены на рисунке {n}.")
hrv_table=table("Показатели ВСР по ЭКГ: медианы окон, прошедших контроль",["ID","ЧСС, уд/мин","SDNN, мс","RMSSD, мс","pNN50, %","ИН, у. е.","VLF, мс²","LF, мс²","HF, мс²","LF/HF","Общая мощность, мс²"],[
 [f"P{i+1:02}",summary(pid,"hr_bpm"),summary(pid,"sdnn_ms"),summary(pid,"rmssd_ms"),summary(pid,"pnn50_pct"),summary(pid,"stress_index",0),
  summary(pid,"vlf_ms2",0),summary(pid,"lf_ms2",0),summary(pid,"hf_ms2",0),summary(pid,"lf_hf",2),summary(pid,"total_ms2",0)]
 for i,pid in enumerate(order)],[34,44,41,44,41,40,44,44,42,38,83])
add(f"ИН — индекс напряжения Баевского {cite('baevsky')}: AMo / (2 × Mo × MxDMn), где Mo — центр модального интервала гистограммы RR с шагом 50 мс, с; AMo — доля RR в этом интервале, %; MxDMn — размах RR, с. Рассчитывается при не менее 30 принятых RR в окне. Временные показатели и ИН — медианы окон с ecg_quality = usable_automatic; спектральные — медианы тех же окон с рассчитанным спектром. ULF не рассчитывается: окно 120 с короче периода ULF (> 300 с).","SmallRu")
table("Частота дыхания по поясу RIP",["Участник","Окон с периодическим дыханием","ЧДД RIP, /мин"],[
 [f"P{i+1:02}",str(summaries[pid].get("rip_windows","н/д")),summary(pid,"rip_rate_bpm")] for i,pid in enumerate(order)],[80,250,165])
add("ЧДД RIP — медиана окон 120 с со статусом RIP_periodic_candidate (доминирующая частота пояса RIP с достаточной периодичностью).","SmallRu")
add(f"В {gsr_review_count} из {len(bio)} записей GSR обнаружены плоские участки или значения вне рабочего диапазона. Эти данные сохранены с отметкой качества.")
if ppg_records:
    subheading("Контактный ФПГ и ЭКГ")
    add("Контактный ФПГ записан вместе с ЭКГ одним прибором. Сравнение использует общую шкалу отсчётов БиоРадио. Пульсовые интервалы рассчитаны после фильтрации и контроля качества.")
    table("Сравнение контактного ФПГ и ЭКГ",["Участник","Окон ФПГ / прошли контроль","Сравнено окон","MAE ЧСС, уд/мин"],[
        [names.get(b["subject_id"],b["subject_id"]),
         f"{b['ppg_analysis'].get('window_count','—')} / {b['ppg_analysis'].get('usable_ppg_windows','—')}",
         str(b.get("ecg_ppg_comparison",{}).get("paired_windows","—")),
         num(b.get("ecg_ppg_comparison",{}).get("hr_mae_bpm"),2)] for b in ppg_records],[135,130,110,120])
    for b in ppg_records:
        c=b.get("ecg_ppg_comparison",{})
        add(f"{names.get(b['subject_id'],b['subject_id'])}: средняя разность ЧСС ФПГ − ЭКГ составляет {num(c.get('hr_bias_ppg_minus_ecg_bpm'),2)} уд/мин. Средняя абсолютная разность RMSSD — {num(c.get('rmssd_mae_ms'),2)} мс, SDNN — {num(c.get('sdnn_mae_ms'),2)} мс. В сравнение включены окна, прошедшие контроль обоих каналов.")
        figure(OUT/"bio"/b["subject_id"]/"ecg_ppg_comparison.png","Контактный ФПГ, R-зубцы ЭКГ и сравнение частоты пульса на общей шкале прибора",intro="Фрагмент сигналов и ЧСС по окнам показаны на рисунке {n}.")
else:
    add("Контактный ФПГ в исходных записях отсутствует; пульс по видео сопоставлен с ЭКГ.")
story.append(PageBreak())
heading("Видеоаналитика")
subheading("Лицевые области, видеопульс и дыхание")
add(f"По точкам позы выделена область головы, затем Face Landmarker определил лицевые ориентиры {cite('mediapipe')}. Для rPPG использованы лоб и щёки. Рот и глаза отслежены отдельно. Размер кадра, частота и соотношение сторон сохранены для каждой записи.")
add(f"Видеопульс рассчитан методом POS {cite('pos')}, с фильтрацией 0,7–3,5 Гц. Частота оценена в окнах по 30 с с шагом 5 с. Спектральная концентрация использована для отбора окон. По положению головы, плеч, кистей, таза и видимых стоп рассчитаны двумерные координаты, скорости и углы; по колебаниям плеч — частота дыхательного движения.")
rows=[]
for i,pid in enumerate(order):
    v=by_video.get(pid,{})
    c=by_alignment.get(pid,{}).get("crossmodal_summary",{})
    rows.append([f"P{i+1:02}",percent(v.get("face_coverage")),
       f"{len(v.get('windows',[]))} / {sum(w.get('rppg_hr_bpm') is not None for w in v.get('windows',[]))} / {c.get('hr_paired_windows',0)}",
       num(c.get("hr_mae_bpm"),2),f"{num(c.get('resp_mae_bpm'),2)} (n={c.get('resp_paired_windows',0)})"])
table("Покрытие лица, видеопульс и дыхательное движение",["Участник","Кадры с ROI","Окна: всего / ЧСС / пары ЭКГ","MAE ЧСС, уд/мин","MAE дыхания, /мин"],rows,[60,85,140,95,115])
add(f"MAE — средняя абсолютная разность частот: видеопульс сопоставлен с ЭКГ, движение плеч — с RIP. Использованы автоматические смещения из таблицы {sync_table} и окна с допустимым качеством; n — число пар дыхательных оценок.","SmallRu")
agreement=report_figures.bland_altman([(codes[a["participant_id"]],pd.read_csv(OUT/"sync"/f"{a['recording_id']}.comparisons.csv"))
    for a in audios if (OUT/"sync"/f"{a['recording_id']}.comparisons.csv").exists()],FIGURES/"rppg_bland_altman.png")
if agreement["pairs"]:
    figure(FIGURES/"rppg_bland_altman.png","Согласие ЧСС rPPG и ЭКГ по окнам 30 с (метод Бланда — Альтмана). Сплошная линия — средняя разность, пунктир — ±1,96 SD",max_height=300,
           intro=f"По {counted(agreement['pairs'],'паре','парам','парам')} окон средняя разность ЧСС rPPG − ЭКГ составила {num(agreement['bias_bpm'],2)} уд/мин, пределы согласия ±1,96 SD — от {num(agreement['low_bpm'],1)} до {num(agreement['high_bpm'],1)} уд/мин; разность не более 5 уд/мин — в {percent(agreement['within_5_bpm'])} окон (рисунок {{n}}). Окна перекрываются с шагом 5 с и не являются независимыми.")
figure(OUT/"audit/video_quality.png","Доля кадров с лицевыми областями и доля окон с рассчитанной ЧСС rPPG",intro="Покрытие лицевыми областями и доля окон с оценкой ЧСС по записям показаны на рисунке {n}.")
subheading("Кинематика тела, головы и конечностей")
add(f"Точки тела получены MediaPipe Pose Landmarker {cite('mediapipe')} с частотой до 5 Гц, лицевые ориентиры — с частотой анализа 12,5 Гц. Точка учитывается при видимости > 0,6. Часть тела без видимых точек пропускается; расчёт остальных показателей продолжается. Координаты нормированы на размер кадра.")
for item in ["наклон оси корпуса — угол отрезка от середины таза к середине плеч относительно вертикали, град; положительный — наклон вправо в кадре;",
             "смещение центра корпуса — межквартильный размах горизонтальной координаты середины плеч и таза, % ширины кадра;",
             "кивки и повороты головы — колебания вертикальной и горизонтальной координаты кончика носа в полосе 0,3–3 Гц: частота максимума спектра Уэлча скорости и среднеквадратичная амплитуда смещения, % кадра; используются непрерывные участки с лицом не короче 20 с;",
             "наклон головы вбок — межквартильный размах угла линии внешних углов глаз, град;",
             "скорость кистей — медиана модуля скорости середины запястий, % кадра/с;",
             "кисть у лица (самоадаптер) — доля кадров позы, в которых запястье ближе к носу, чем 0,8 ширины плеч;",
             "скрещённые руки или ноги — доля кадров, в которых оба запястья (обе лодыжки) находятся по разные стороны от средней линии плеч (таза), противоположные своей стороне; при менее чем 10 кадрах с видимыми точками доля не рассчитывается."]:
    add("– "+item,"ListRu")
table("Поза и корпус",["Участник","Видимость стоп","Наклон корпуса: медиана, град","Наклон корпуса: IQR, град","Смещение центра, %","Скрещённые руки, %","Скрещённые ноги, %"],[
 [f"P{i+1:02}",percent(summaries[pid].get("feet_visible_fraction")),summary(pid,"torso_tilt_median_deg"),summary(pid,"torso_tilt_iqr_deg"),
  summary(pid,"centre_shift_iqr",2,100),summary(pid,"crossed_arms_fraction",1,100),summary(pid,"crossed_legs_fraction",1,100)]
 for i,pid in enumerate(order)],[55,65,80,75,70,75,75])
table("Голова и кисти",["ID","Кивки: частота, Гц","Кивки: амплитуда, %","Повороты: частота, Гц","Повороты: амплитуда, %","Наклон вбок: IQR, град","Скорость кистей, %/с","Кисть у лица, %"],[
 [f"P{i+1:02}",summary(pid,"nod_frequency_hz",2),summary(pid,"nod_rms",2,100),summary(pid,"turn_frequency_hz",2),summary(pid,"turn_rms",2,100),
  summary(pid,"head_roll_iqr_deg"),summary(pid,"hand_speed_median",2,100),summary(pid,"hand_to_face_fraction",1,100)]
 for i,pid in enumerate(order)],[52,60,62,62,64,62,63,70])
story.append(PageBreak())
heading("Речевой анализ")
subheading("Распознавание речи и проверка стенограмм")
models_used=", ".join(sorted({str(a.get("model","Whisper")) for a in audios}))
add(f"Расшифровка получена моделью {models_used} {cite('whisper')} через faster-whisper {cite('faster_whisper')}; речевые интервалы выделены Silero VAD {cite('silero')}. При вычитке проверены отрицания, короткие ответы и вопросы; сохранены временные метки слов и сегментов.")
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
subheading("Речевые и акустические показатели")
add(f"По аудиодорожкам извлечены основной тон и его изменчивость, энергия, паузы, доля звучащей речи, спектральный центр, наклон и форманты; основной тон и форманты рассчитаны Praat через Parselmouth {cite('parselmouth')}. Для пар «вопрос – ответ» рассчитаны задержка и длительность ответа, темп в словах/мин и оценка слогов/с по числу гласных. Для начального разделения голосов использованы представления WeSpeaker ResNet34 {cite('wespeaker')} и кластеризация KMeans на два диктора.")
add("Латентность — интервал от конца последней реплики вопроса до начала ответа; отрицательные значения (перебивание) исключены. Темп рассчитан по всем репликам респондента: слова и гласные буквы текста, делённые на суммарную длительность реплик. Акустика — медианы окон 5 с с шагом 1 с, центр которых лежит в реплике респондента, при доле вокализованных отсчётов ≥ 0,2.")
table("Временные параметры диалога",["Участник","Пар «вопрос – ответ»","Латентность: медиана, с","Латентность: IQR, с","Длительность ответа: медиана, с","Слов/мин","Слогов/с"],[
 [f"P{i+1:02}",str(summaries[pid].get("qa_pairs","н/д")),summary(pid,"latency_median_s",2),summary(pid,"latency_iqr_s",2),
  summary(pid,"answer_median_s",2),summary(pid,"words_per_min",0),summary(pid,"syllables_per_s",2)]
 for i,pid in enumerate(order)],[55,70,75,70,85,65,75])
table("Акустические характеристики голоса",["ID","F0: медиана, Гц","F0: IQR, Гц","Уровень, dBFS","Центр спектра, Гц","Наклон спектра, дБ/окт","F1, Гц","F2, Гц","F3, Гц"],[
 [f"P{i+1:02}",summary(pid,"pitch_median_hz",0),summary(pid,"pitch_iqr_hz",0),summary(pid,"rms_dbfs"),summary(pid,"spectral_centroid_hz",0),
  summary(pid,"spectral_tilt_db_per_octave",2),summary(pid,"f1_median_hz",0),summary(pid,"f2_median_hz",0),summary(pid,"f3_median_hz",0)]
 for i,pid in enumerate(order)],[50,52,48,52,65,70,50,50,58])
story.append(PageBreak())
heading("Разметка ответов")
subheading("Сопоставление с письменным эталоном")
add("Письменные истории сопоставлены с интервью по событию, действующему лицу, времени, действию и отрицанию. Короткие ответы рассмотрены вместе с предшествующим вопросом.")
table("Категории сверки с письменным эталоном",["Категория","Критерий"],[[relation[k],v] for k,v in [
 ("consistent","Устный ответ воспроизводит событие или деталь эталона."),
 ("contradiction_candidate","Отрицание факта, замена события, примера или существенной детали эталона."),
 ("different","Сведения вне проверяемого факта; замены эталонного события или детали нет."),
 ("omitted","Письменный эпизод не пересказан."),
 ("uncertain","Смысл ответа или отдельные слова остаются неоднозначными.")]], [175,320])
comparison_table=table("Результаты сверки устных ответов с письменными историями",["Результат сверки","Число сопоставлений"],[[relation[k],str(counts[k])] for k in relation],[365,130])
add(f"Всего {counted(len(notes),'запись','записи','записей')} сверки (таблица {comparison_table}). Одна история может содержать несколько проверяемых утверждений. Для найденных ответов сохранены цитаты, вопрос и время на видео; отсутствующие пересказы отмечены отдельно. Ключевые сопоставления по каждому участнику приведены в приложении А, полная таблица с цитатами и таймкодами — в outputs/research/content_comparison.csv.")

story.append(PageBreak())
heading("Статистический анализ")
if not multimodal.get("models"):
    raise SystemExit("Run scripts/analyze_multimodal.py before building the report")
contrast_summary=read(OUT/"research/contrasts/summary.json")
if not contrast_summary:
    raise SystemExit("Run scripts/analyze_contrasts.py before building the report")
def holm_p(value):
    return "н/д" if value is None else num(value,3)
def delta_text(delta):
    text=f"Δ Brier = {num(delta['delta_brier'],4)}; 95 % ДИ [{num(delta['ci_low'],4)}; {num(delta['ci_high'],4)}]"
    return text+("; интервал включает 0" if delta["ci_low"]<=0<=delta["ci_high"] else "")
def significant(rows,key):
    return [row for row in rows if row.get(key) is not None and row[key]<.05]
labels_ru={**FEATURE_LABELS,contrast_lib.ANOMALY:"Норма вектора акустических аномалий"}
tests=contrast_summary["tests"]
contrast_models={row["id"]:row["metrics"] for row in contrast_summary["models"]}
normalization_delta=contrast_summary["increments"]["normalized_minus_raw"]
subheading("Метки ответов и нормализация")
add(f"Каждому ответу присвоена одна из трёх меток: «Правда» — ответ согласуется с письменным эталоном; «Ложь» — ответ содержит несоответствие эталону (раздел {ref('Разметка ответов')}); «База» — ответ на вопрос нейтрального блока (о себе, увлечения, самочувствие; раздел {ref('Этапы сессии')}). Ответ с меткой «Правда» или «Ложь» в «Базу» не входит.")
count_rows=[]
for i,pid in enumerate(order):
    c=next((row for row in contrast_summary["counts"] if row["participant_id"]==pid),{})
    count_rows.append([f"P{i+1:02}",str(c.get("truth",0)),str(c.get("lie",0)),str(c.get("base",0))])
count_rows.append(["Всего",*[str(sum(int(row[j]) for row in count_rows)) for j in (1,2,3)]])
labels_table=table("Число ответов по меткам",["Участник","«Правда»","«Ложь»","«База»"],count_rows,[110,128,128,129])
add(f"Нормализация выполняется отдельно для каждого участника и признака: z = (x − медиана «Базы») / (IQR всех ответов участника / 1,349); z ограничено интервалом [−{num(contrast_lib.Z_LIMIT,0)}; {num(contrast_lib.Z_LIMIT,0)}]. Метки «Правда» и «Ложь» в расчёт центра и масштаба не входят. При менее чем {contrast_lib.MIN_BASE} значениях «Базы» или менее чем {contrast_lib.MIN_SCALE} ответах с измеренным признаком z не рассчитывается. Вектор акустических аномалий ответа — z основного тона, разброса F0, уровня, темпа и доли пауз; его норма — корень из среднего квадрата доступных компонент.")
add("Контраст участника по признаку — среднее z ответов «Ложь» минус среднее z ответов «Правда». Единица статистического вывода — участник.")
subheading("Гипотезы")
group=contrast_summary["group"]
group_codes=", ".join(codes[pid] for pid in order if pid in group["participants"])
other_codes=", ".join(codes[pid] for pid in order if pid not in group["participants"])
estimable=lambda key:sum(row.get(key) is not None for row in tests)
def verdict(rows,key):
    hits=significant(rows,key)
    return "p Холма < 0,05: "+", ".join(labels_ru.get(r["feature"],r["feature"]) for r in hits) if hits else "Нулевая гипотеза не отвергнута ни для одного признака (p Холма ≥ 0,05)"
hypotheses=[
 ["H1","Контраст «Ложь − Правда» отличается от нуля",f"Точный двусторонний тест перестановки знаков по участникам; поправка Холма по {estimable('h1_p')} признакам",verdict(tests,"h1_p_holm")],
 ["H2",f"Контраст различается между группами профиля: ведущий естественный тип «Содействующий» ({group_codes}) и остальные ({other_codes})",f"Точный перестановочный тест разности средних групп; минимум {contrast_lib.MIN_GROUP} участника в группе; поправка Холма по {estimable('h2_p')} признакам",verdict(tests,"h2_p_holm")],
 ["H3","Контраст связан с выраженностью профиля","Коэффициент Спирмена, точные перестановки рангов; минимум 6 участников; поправка Холма",verdict(tests,"h3_p_holm")],
 ["H4","Профили снижают ошибку модели «видео + речь»","Δ Brier с профилями и без, LOPO, bootstrap по участникам",delta_text(multimodal["profile_increment"])],
 ["H5","Нормализация к «Базе» снижает ошибку модели","Δ Brier нормированных и исходных признаков, LOPO, bootstrap по участникам",delta_text(normalization_delta)],
]
hypotheses_table=table("Проверяемые гипотезы, уровень значимости 0,05",["№","Гипотеза H1 (альтернатива)","Метод","Результат"],hypotheses,[28,150,160,157])
subheading("Контрасты и маркеры по группам профиля")
add(f"Контрасты по признакам и группам приведены в таблице {numbers['table']+1}. Признак не рассчитывается, если контраст доступен менее чем у 6 участников (основной тон, его разброс и уровень речи вычисляются только по окнам 5 с внутри ответа).")
marker_rows=[]
for row in tests:
    marker_rows.append([labels_ru.get(row["feature"],row["feature"]),str(row["h1_n"]),
        f"{num(row['h1_mean'],2)} [{num(row['h1_ci_low'],2)}; {num(row['h1_ci_high'],2)}]" if row["h1_mean"] is not None else "н/д",holm_p(row.get("h1_p_holm")),
        num(row["h2_group_mean"],2),num(row["h2_other_mean"],2),holm_p(row.get("h2_p_holm")),
        num(row["h3_rho"],2),holm_p(row.get("h3_p_holm"))])
table("Контраст «Ложь − Правда» в единицах z: все участники, группы профиля и связь с выраженностью",
      ["Признак","n","Среднее [95 % ДИ]","p Холма H1","Сод.","Др.","p Холма H2","ρ","p Холма H3"],marker_rows,[108,22,92,48,40,40,48,40,57])
add("n — число участников с контрастом; Сод. и Др. — средний контраст группы с ведущим естественным типом «Содействующий» и остальных участников; ρ — коэффициент Спирмена контраста и выраженности профиля.","SmallRu")
ranked=sorted([row for row in tests if row["h2_group_mean"] is not None],key=lambda row:-abs(row["h2_group_mean"]))[:3]
ranked_other=sorted([row for row in tests if row["h2_other_mean"] is not None],key=lambda row:-abs(row["h2_other_mean"]))[:3]
if ranked and ranked_other:
    add("Наибольшие по модулю средние контрасты группы «Содействующий»: "+"; ".join(f"{labels_ru.get(r['feature'],r['feature']).split(',')[0].lower()} ({num(r['h2_group_mean'],2)})" for r in ranked)+". Группы остальных профилей: "+"; ".join(f"{labels_ru.get(r['feature'],r['feature']).split(',')[0].lower()} ({num(r['h2_other_mean'],2)})" for r in ranked_other)+". Значимость этих различий приведена в столбцах p Холма.")
table("Модели LOPO на исходных и нормированных к «Базе» признаках",["Модель","Brier","95 % ДИ Brier","Сбаланс. точность","ROC AUC"],[
 [row["name"],num(row["metrics"]["brier"],3),f"[{num(row['metrics']['brier_ci_low'],3)}; {num(row['metrics']['brier_ci_high'],3)}]",
  num(row["metrics"]["balanced_accuracy"],3),num(row["metrics"]["roc_auc"],3)] for row in contrast_summary["models"]],[185,50,100,85,75])
add(f"Признаки: речь (5), видео (4), ЧСС и RMSSD ЭКГ в окне 120 с. Модель, разбиение и метрики совпадают с разделом {ref('Модели классификации')}. Нормированные признаки рассчитываются по ответам исключённого участника без использования меток.","SmallRu")
subheading("Модели классификации")
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
add(f"Проверка LOPO каждый раз исключает из обучения целиком одного участника {cite('lopo')}. Заполнение пропусков и масштабирование обучаются только на остальных. L2-логистическая регрессия: C = 0,1, порог 0,5, без подбора параметров. Участники имеют равный вес при обучении и расчёте метрик.","SmallRu")
add("Речевые признаки: основной тон, его разброс, уровень сигнала, темп и доля пауз. Видеопризнаки: скорость головы, движение торса, раскрытие глаз и движение рта. БиоРадио: ЧСС, RMSSD ЭКГ и частота RIP в 120-секундном контексте ответа.","SmallRu")
add(f"До заполнения пропусков основной тон доступен для {feature_counts['audio_pitch_median_hz']} ответов, движение головы — для {feature_counts['video_head_speed_norm_per_s']}, рта — для {feature_counts['video_mouth_motion_ratio_per_s']}, ЧСС ЭКГ — для {feature_counts['bio_ecg_hr_bpm_context120s']}, частота RIP — для {feature_counts['bio_rip_rate_bpm_context120s']}.","SmallRu")
add("Профили включают выраженность, Эмоц+, Эмоц− и отдельные признаки наличия компонентов естественного и демонстрируемого типа. Эмоциональный баланс используется в корреляциях. Имена, ИСУ и тексты вопросов и эталонов в признаки моделей не входят.","SmallRu")
table("Качество моделей при проверке LOPO",["Модель","Brier","95 % ДИ Brier","Сбаланс. точность","ROC AUC"],[
 [model_name(row["name"]),num(row["metrics"].get("brier"),3),
  f"[{num(row['metrics'].get('brier_ci_low'),3)}; {num(row['metrics'].get('brier_ci_high'),3)}]",
  num(row["metrics"].get("balanced_accuracy"),3),num(row["metrics"].get("roc_auc"),3)]
 for row in model_rows],[185,50,100,85,75])
add(f"Brier {cite('brier')} — средняя квадратичная ошибка вероятности; меньше лучше. Сбалансированная точность — среднее чувствительности и специфичности; ROC AUC — качество ранжирования ответов. Для последних двух метрик случайный уровень равен 0,5. 95 % ДИ рассчитаны bootstrap по участникам ({increment.get('bootstrap',5000)} повторов) при фиксированных прогнозах LOPO, без повторного обучения.","SmallRu")
add(f"Среди проверенных наборов наименьшая ошибка у видеопризнаков: Brier {num(model_metrics['video']['brier'],3)} против {num(model_metrics['prior']['brier'],3)} у прогноза по доле классов. Однако ROC AUC = {num(model_metrics['video']['roc_auc'],3)}, а сбалансированная точность = {percent(model_metrics['video']['balanced_accuracy'])}: разделение ответов осталось около случайного уровня.","SmallRu")

subheading("Вклад профилей и биосигналов")
fig=OUT/"research/multimodal/model_comparison.png"
if not fig.exists():raise SystemExit(f"Missing analysis figure: {fig}")
figure(fig,"Ошибка на исключённых участниках и изменение при добавлении профилей. Ромбы показывают средние; отдельные точки отражают участников",intro="Ошибка моделей по участникам и изменение ошибки при добавлении профилей показаны на рисунке {n}.")
add(f"Добавление профилей к видео и речи изменило Brier на {num(increment['delta_brier'],4)}; 95 % ДИ [{num(increment['ci_low'],4)}; {num(increment['ci_high'],4)}]. Положительная разность означает рост ошибки. Интервал включает ноль; преимущество профилей не установлено.","SmallRu")
classified=model_metrics['av_profiles']['confusion_counts']
add(f"Модель «видео + речь + профили» обнаружила {classified['tp']} из {dataset['contradictions']} {plural(dataset['contradictions'],'несоответствия','несоответствий','несоответствий')}, пропустила {classified['fn']} и дала {counted(classified['fp'],'ложное срабатывание','ложных срабатывания','ложных срабатываний')}. Добавление БиоРадио снизило Brier с {num(model_metrics['av']['brier'],3)} до {num(model_metrics['fusion']['brier'],3)}, но ROC AUC осталось {num(model_metrics['fusion']['roc_auc'],3)}. Объединение каналов не обеспечило распознавания.","SmallRu")
add("Проверена чувствительность результатов к временному совмещению: расчёт повторён на нижней и верхней границах интервалов автоматического смещения при scale = 1.","SmallRu")
table("Чувствительность моделей к границам смещения",["Граница смещения","Модель","Brier","ROC AUC"],[
 [scenario["name"],model_name(row["name"]),
  num(row["metrics"].get("brier"),3),num(row["metrics"].get("roc_auc"),3)]
 for scenario in multimodal["sync_sensitivity"] for row in scenario["models"]],[145,170,90,90])
add(f"При обоих смещениях ROC AUC остаётся ниже 0,5. Для полного набора каналов добавление профилей даёт Δ Brier = {num(fusion_increment['delta_brier'],6)}, 95 % ДИ [{num(fusion_increment['ci_low'],4)}; {num(fusion_increment['ci_high'],4)}]; улучшение также не установлено.","SmallRu")

subheading("Связи профилей с измерениями")
families=multimodal["association_families"]
associations=multimodal["associations"]
estimated=sum(row.get("rho") is not None for row in associations)
add(f"Сопоставлены профили и показатели за всю запись. Единица анализа — участник, n = {dataset['participants']}. Заданы {families['primary']} основных сочетаний (выраженность и эмоциональный баланс с 6 признаками) и {families['secondary']} дополнительных. Коэффициент Спирмена, точные двусторонние перестановки рангов, поправка Холма {cite('holm')} отдельно в каждом семействе; 95 % ДИ — парный bootstrap участников {cite('permutation')}.","SmallRu")
add(f"Рассчитаны {estimated} из {len(associations)} сочетаний; остальные не набрали минимум 6 пригодных участников или вариативных значений. Для ЭКГ, RIP и rPPG требуется не менее 20 % пригодных окон записи. После поправки Холма статистически значимых связей не обнаружено (p < 0,05).","SmallRu")
fig=OUT/"research/multimodal/associations.png"
if not fig.exists():raise SystemExit(f"Missing analysis figure: {fig}")
figure(fig,f"Все {families['primary']} основных сочетаний. Одна точка соответствует одному участнику; подписи содержат n, коэффициент Спирмена и p Холма",max_height=540,intro="Все основные сочетания показаны на рисунке {n}.")
head_association=next(row for row in associations if row['profile']=='expression' and row['feature']=='video_head_speed_norm_per_s')
add(f"Самая сильная по модулю связь в основном семействе — выраженность профиля и скорость головы: ρ = {num(head_association['rho'],3)}, p Холма = {num(head_association['p_holm'],3)}. Более высокой выраженности соответствовала меньшая скорость движения головы; связь не достигла статистической значимости. Здесь сравниваются различия между участниками за всю запись; связь этого признака с меткой ответа отдельно не установлена.","SmallRu")

story.append(PageBreak())
heading("Интерфейс")
add("Локальное веб-приложение (FastAPI, nti_lab.app) показывает запись участника в пяти синхронизированных зонах. Все зоны обновляются по одному курсору времени видео; биосигналы переводятся в шкалу видео сохранённым смещением.")
table("Зоны интерфейса",["Зона","Содержимое"],[
 ["1. Временная шкала","Ползунок времени сессии, воспроизведение и покадровый шаг; цветные интервалы начального и конечного фона и меток «Правда», «Ложь», «База», «Вопрос», «Ответ», «Тема»; переход к интервалу по щелчку"],
 ["2. Расшифровка","Реплики с таймкодами, интервьюер и респондент выделены цветом; текущая реплика подсвечивается; рядом — письменная история и цитата сопоставления с переходом к ответу"],
 ["3. Биосигналы","ЭКГ, rPPG, дыхание RIP, GSR и контактный ФПГ при наличии; окно 20 с с курсором времени"],
 ["4. Медиа","Видео с наложением областей лица (лоб, щёки, рот, глаза) и скелета позы"],
 ["5. Аналитика","Спектр ВСР окна 120 с; ЧСС ЭКГ и rPPG, RMSSD, индекс напряжения Баевского, основной тон, темп сегмента, задержка ответа, LF/HF, частота дыхания; сохранение смещения БиоРадио и меток"]],[110,385])
figure(OUT/"report/interface.png","Интерфейс просмотра и сопоставления записей",max_height=340,intro="Окно приложения показано на рисунке {n}.")
add("Запуск: start.cmd. Установка для Windows/Linux/macOS и запуск через Docker описаны в README.")
story.append(PageBreak())
heading("Выводы по работе",numbered=False)
add("Создан комплекс обработки и синхронного просмотра видео, речи и БиоРадио. Рассчитаны характеристики голоса, движений, пульса, дыхания и вариабельности сердечного ритма. Сопоставление устных ответов с письменными историями выявило несоответствия у каждого участника; для них сохранены проверяемые цитаты и временные границы.")
add("Автоматическое распознавание лжи по измеренным признакам на этих данных не достигнуто. Лучший из проверенных наборов — движения головы и торса, раскрытие глаз и движение рта — дал качество около случайного уровня. Речевые признаки, биосигналы и их объединение не обеспечили улучшения относительно простого прогноза по доле классов. Отдельный надёжный маркер лжи среди рассчитанных показателей не установлен.")
add("Добавление профилей участников не снизило ошибку вероятностного прогноза Brier. Корреляционный анализ профилей с показателями за всю запись также не выявил значимых связей после поправки на множественные проверки.")
add("Сравнение контактных каналов показало близкие оценки ЧСС по ФПГ и ЭКГ в прошедших контроль окнах. Точность видеопульса заметно различалась между записями. Для классификации основную полноту данных обеспечивала кинематика; акустические и биосигнальные признаки содержали пропуски.")
hits=[r for key in ("h1_p_holm","h2_p_holm","h3_p_holm") for r in significant(tests,key)]
add(("После нормализации к «Базе» ни один признак не показал значимого контраста «Ложь − Правда», различия между группами профиля или связи с выраженностью (p Холма ≥ 0,05)." if not hits else
     "После нормализации к «Базе» p Холма < 0,05 получено для: "+", ".join(sorted({labels_ru.get(r["feature"],r["feature"]) for r in hits}))+".")
    +f" Нормализация к «Базе» изменила ошибку модели на {delta_text(normalization_delta).removeprefix('Δ Brier = ')}.")

story.append(PageBreak())
heading("Список использованных источников",numbered=False)
for i,key in enumerate(cited,1):
    add(f"{i}. {REFERENCES[key]}","SmallRu")

story.append(PageBreak())
heading("Приложение А. Результаты сверки историй по участникам",numbered=False)
add(f"Результаты сверки историй каждого участника приведены в таблицах {numbers['table']+1}–{numbers['table']+len(order)}; после таблицы — ключевые сопоставления с цитатами и таймкодами видео.")
for i,pid in enumerate(order):
    if i:story.append(PageBreak())
    contents_entry(names.get(pid,pid),"Sub",1)
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
heading("Приложение Б. Файлы результатов",numbered=False)
table("Файлы результатов",["Путь в проекте","Содержимое"],[
 ["outputs/research/content_comparison.csv","Сопоставления с цитатами, контекстом и временем"],
 ["outputs/research/multimodal/analysis-results.zip","Полный анализ: CSV, графики, план, происхождение, параметры и код"],
 ["outputs/research/multimodal/associations.csv","Все корреляции: n, rho, p, поправка Холма и интервалы"],
 ["outputs/research/multimodal/participant_features.csv","Показатели участников за всю запись"],
 ["outputs/research/multimodal/trial_features.csv","Показатели отдельных ответов"],
 ["outputs/research/multimodal/model_predictions.csv","Прогнозы LOPO для воспроизведения оценок"],
 ["outputs/research/contrasts/","Ответы с метками «Правда», «Ложь», «База», нормированные признаки, контрасты участников и проверка гипотез"],
 ["outputs/report/figures/","Рисунки синхронизации, согласия rPPG и ЭКГ и синхронных сигналов"],
 ["outputs/report/session_summary.csv",f"Сводные показатели этапов сессии, ВСР, речи и кинематики по записям (разделы {ref('Этапы сессии')}, {ref('Биосигналы БиоРадио')}–{ref('Речевой анализ')})"],
 ["outputs/audio/","Стенограммы, временные метки, акустические признаки"],
 ["outputs/bio/","ЭКГ, ФПГ, RIP, GSR, интервалы и показатели ВСР"],
 ["outputs/video/","Лицевые области, rPPG и кинематика"],
 ["outputs/sync/","Фоновые участки и оценки смещения"],
 ["data/private/","Письменные эталоны и каталог источников"],
 ["docs/validation.md","Порядок технических проверок"],
],[235,260])

def footer(c,doc):
    c.setFont("Body",8);c.setFillColor(colors.HexColor(MUTED))
    c.drawRightString(A4[0]-50,28,str(doc.page))

class ReportDocument(SimpleDocTemplate):
    def afterFlowable(self,flowable):
        level=getattr(flowable,"toc_level",None)
        if level is None:return
        text=flowable.getPlainText();key=f"section-{id(flowable)}"
        self.canv.bookmarkPage(key);self.canv.addOutlineEntry(text,key,level=level)
        self.notify("TOCEntry",(level,text,self.page,key))

path=args.output
path.parent.mkdir(parents=True,exist_ok=True)
doc=ReportDocument(str(path),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=42,bottomMargin=45,
                   title=TITLE,author=", ".join(name for name,_ in authors))
doc.multiBuild(story,onFirstPage=lambda c,doc:None,onLaterPages=footer)
print(path)
