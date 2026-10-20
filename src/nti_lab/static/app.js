const $=id=>document.getElementById(id);
let D={},list=[],sid="",current=0,signals={},signalAt=-999,lastRefresh=0,generation=0,signalGeneration=0;
const video=$("video");
const labelText={truth:"Правда",lie:"Ложь",baseline:"База",question:"Вопрос",answer:"Ответ",topic:"Тема",unknown:"Неизвестно"};
const clock=t=>`${Math.floor(t/60).toString().padStart(2,"0")}:${(t%60).toFixed(2).padStart(5,"0")}`;
const fmt=(n,d=1)=>n!==null&&n!==undefined&&n!==""&&Number.isFinite(Number(n))?Number(n).toFixed(d):"—";
const near=(a,key,t)=>a?.length?a.reduce((best,r)=>Math.abs(Number(r[key])-t)<Math.abs(Number(best[key])-t)?r:best,a[0]):{};
function node(tag,text,cls){const e=document.createElement(tag);e.textContent=text;if(cls)e.className=cls;return e}
async function json(url,opts){const r=await fetch(url,opts);if(!r.ok)throw Error(`${r.status}: ${await r.text()}`);return r.json()}
function notify(s){$("notice").textContent=s}
async function init(){
 list=await json("/api/sessions");
 $("sessions").replaceChildren(...list.map(s=>{let o=node("option",s.id.replace("_"," · "));o.value=s.id;return o}));

 if(list.length)await load(list[0].id);
 else $("status").textContent="Нет записей. Запустите обработку данных по README.";
}
async function load(id){
 const requestGeneration=++generation;
 video.pause();sid=id;current=0;signals={};signalAt=-999;
 $("status").textContent="Загрузка сессии…";
 const response=await json(`/api/session/${id}`);
 if(requestGeneration!==generation)return;
 D=response;
 $("sessions").value=sid;
 if(D.media_ready){video.src=`/media/${sid}`;video.load()}else{video.removeAttribute("src");video.load()}
 $("time").max=D.video.duration_s||D.audio.duration_s||D.bio.analysis_duration_s||1200;
 $("offset").value=D.annotation.bio_offset_s||0;$("scale").value=D.annotation.bio_scale||1;
 $("verified").checked=!!D.annotation.sync_verified;
 const alignment=D.alignment||{},bounds=alignment.offset_uncertainty_interval_s;
 const states={supported_unverified:"Согласующийся кандидат",ambiguous_unverified:"Неоднозначная оценка",insufficient_signal:"Недостаточно сигнала"};
 $("alignmentInfo").textContent=alignment.status?`${states[alignment.status]||alignment.status}. Предложенное смещение: ${fmt(alignment.suggested_offset_s,3)} с.${bounds?` Диапазон неопределённости: ${fmt(bounds[0],3)}…${fmt(bounds[1],3)} с.`:""} Требуется проверка общих событий.`:"Автоматическая оценка ещё не рассчитана.";
 $("baselines").replaceChildren(...(alignment.baselines||[]).map(b=>{const button=node("button",`${b.name==="beginning"?"Начальный":"Конечный"} фон · ${clock(b.start_s)}–${clock(b.end_s)}`);button.onclick=()=>seek(b.start_s);return button}));

 $("status").textContent=`${D.media_ready?"Видео готово":"Видео обрабатывается"} · ${D.audio.segments?.length||0} речевых сегментов · ${D.bio.channels?.length||0} каналов БОС`;
 $("asrInfo").textContent=`${D.audio.model||"ASR обрабатывается"} · автоматическая расшифровка`;
 $("faceCoverage").textContent=`Лицо отслежено: ${fmt((D.video.face_coverage??NaN)*100,1)}% кадров`;
 renderTranscript();renderStories();renderEvents();renderRoles();update(0);
}
function renderTranscript(){
 const segs=D.audio.segments||[];
 $("transcript").replaceChildren(...segs.map((s,i)=>{
  const b=node("button","");b.dataset.index=i;b.dataset.speaker=s.speaker||"unknown";
  b.append(node("span",clock(s.start),"time"),node("span",s.speaker||s.speaker_id||"Голос","speaker"),node("span",s.text));
  b.onclick=()=>seek(s.start);return b;
 }));
 if(!segs.length)$("transcript").textContent="Автоматическая расшифровка ещё не рассчитана. Нажмите «Обновить данные» после обработки.";
}
function renderStories(){
 $("storySelect").replaceChildren(...D.stories.map(s=>{const o=node("option",s.title||s.topic||s.story_id);o.value=s.story_id;return o}));
 showStory();
}
function showStory(){
 const s=D.stories.find(x=>x.story_id===$("storySelect").value);
 $("story").textContent=s?.text||"Исходная история не связана с участником.";
 const match=D.matches.find(x=>x.candidate_story_id===s?.story_id);
 $("match").replaceChildren();
 if(s?.source_url){const a=node("a","Открыть источник");a.href=s.source_url;a.target="_blank";a.rel="noopener";$("match").append(a)}
 if(match)$("match").append(node("p",`Лексическое сходство: ${fmt(match.similarity_score||match.similarity,3)}. Требуется проверка эпизода.`));
 const names={consistent:"Согласуется с эталоном",contradiction_candidate:"Возможное противоречие",different:"Другой пример или деталь",omitted:"Эпизод не рассказан",uncertain:"Неоднозначный ответ"};
 $("evidence").replaceChildren(...(D.comparisons||[]).filter(x=>x.story_id===s?.story_id).map(r=>{
  const box=node("div","",`finding ${r.relation}`),head=node("strong",names[r.relation]||r.relation);
  box.append(head);
  if(r.start_s!==null){const b=node("button",`${clock(r.start_s)} → слушать с вопросом`);b.onclick=()=>seek(r.context_start_s??r.start_s);box.append(b)}
  box.append(node("p",`Эталон: «${r.source_excerpt||r.source_detail||""}»`));
  if(r.spoken_excerpt)box.append(node("p",`Расшифровка: «${r.spoken_excerpt}»`));
  box.append(node("p",r.interpretation||r.reason||""));
  const check=(D.audio_rechecks||[]).find(c=>c.comparison_id===r.comparison_id);
  if(check){
   box.append(node("p",`Повторная автоматическая ASR: ${check.assessment_note||""}`));
   if(check.quote_change!=="unchanged"&&check.recommended_stable_excerpt)box.append(node("p",`Устойчивый фрагмент повторной расшифровки: «${check.recommended_stable_excerpt}»`));
  }
  if(r.context_excerpt){const details=document.createElement("details");details.append(node("summary","Контекст вопроса и ответа"),node("p",r.context_excerpt));box.append(details)}
  if(!r.audio_verified)box.append(node("small","Вывод по тексту ASR. Перед подтверждением метки проверьте речь и контекст."));
  return box;
 }));
}
function renderRoles(){
 const speakers=[...new Set((D.audio.segments||[]).map(s=>s.speaker||s.speaker_id).filter(Boolean))];
 $("roles").replaceChildren(...speakers.map(s=>{
  const l=node("label",s),sel=document.createElement("select");
  [["unknown","Не назначена"],["interviewer","Интервьюер"],["respondent","Респондент"]].forEach(([v,t])=>{const o=node("option",t);o.value=v;sel.append(o)});
  sel.value=D.annotation.speaker_roles?.[s]||"unknown";
  sel.onchange=async()=>{D.annotation.speaker_roles??={};D.annotation.speaker_roles[s]=sel.value;await save();update(current)};
  l.append(sel);return l;
 }));
}
function renderEvents(){
 $("markers").replaceChildren();$("events").replaceChildren();
 for(const e of D.alignment?.baselines||[]){
  const b=node("button","","marker baseline automatic");b.style.left=`${e.start_s/Number($("time").max)*100}%`;b.style.width=`${(e.end_s-e.start_s)/Number($("time").max)*100}%`;b.title=`Предварительный фон ${clock(e.start_s)}–${clock(e.end_s)}`;b.onclick=()=>seek(e.start_s);$("markers").append(b);
 }
 for(const [i,e] of (D.annotation.events||[]).entries()){
  const b=node("button","",`marker ${e.label}`);b.style.left=`${e.start/Number($("time").max)*100}%`;b.style.width=`${(e.end-e.start)/Number($("time").max)*100}%`;b.title=`${labelText[e.label]} ${clock(e.start)}: ${e.note||""}`;b.onclick=()=>seek(e.start);$("markers").append(b);
  const row=node("div",`${clock(e.start)}–${clock(e.end)} · ${labelText[e.label]} · ${e.verified?"проверено":"предварительно"} · ${e.note||""}`);
  const del=node("button","Удалить");del.onclick=async()=>{D.annotation.events.splice(i,1);await save();renderEvents()};row.append(del);$("events").append(row);
 }
}
async function save(){await json(`/api/session/${sid}/annotation`,{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify(D.annotation)});notify("Разметка сохранена локально")}
function seek(t){current=Math.max(0,Math.min(Number($("time").max),t));if(D.media_ready)video.currentTime=current;update(current)}
function plot(id,traces,t,span){
 const c=$(id),w=c.clientWidth||800,h=Number(c.dataset.logicalHeight||240),dpr=devicePixelRatio||1;
 c.style.height=`${h}px`;
 c.width=w*dpr;c.height=h*dpr;const ctx=c.getContext("2d");ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
 const row=h/Math.max(traces.length,1),left=78,range=span||20,x0=t-range/2;
 traces.forEach((s,i)=>{
  const top=i*row;ctx.strokeStyle="#e9eeea";ctx.beginPath();ctx.moveTo(left,top+row/2);ctx.lineTo(w,top+row/2);ctx.stroke();
  ctx.font="10px Segoe UI";ctx.fillStyle=s.color;ctx.fillText(s.name,4,top+row/2);
  let pairs=(s.t||[]).map((x,j)=>[Number(x),s.y?.[j]]).filter(p=>p[0]>=x0&&p[0]<=x0+range);
  const vals=pairs.map(x=>x[1]).filter(x=>x!==null&&Number.isFinite(Number(x))).map(Number);
  if(!vals.length){ctx.fillStyle="#8b9993";ctx.fillText("нет валидных данных",left+8,top+row/2);return}
  let lo=Math.min(...vals),hi=Math.max(...vals),amp=Math.max(hi-lo,1e-8);
  ctx.strokeStyle=s.color;ctx.lineWidth=1.1;ctx.beginPath();let pen=false;
  pairs.forEach(([x,y])=>{if(y===null||!Number.isFinite(Number(y))){pen=false;return}const px=left+(x-x0)/range*(w-left),py=top+row*.85-(y-lo)/amp*row*.7;if(pen)ctx.lineTo(px,py);else{ctx.moveTo(px,py);pen=true}});ctx.stroke();
 });
 ctx.strokeStyle="#172e3060";ctx.beginPath();ctx.moveTo(left+(w-left)/2,0);ctx.lineTo(left+(w-left)/2,h);ctx.stroke();
}
async function fetchSignal(t){
 if(Math.abs(t-signalAt)<2)return;
 signalAt=t;
 const requestGeneration=generation,requestSignal=++signalGeneration,requestedSid=sid;
 try{const response=await json(`/api/session/${requestedSid}/signal-window?t=${encodeURIComponent(t)}`);
  if(requestGeneration!==generation||requestSignal!==signalGeneration||requestedSid!==sid)return;
  signals=response;drawSignals();drawSpectrum()
 }catch(e){if(requestGeneration===generation)notify(e.message)}
}
function drawSignals(){
 const a=D.annotation||{},off=Number(a.bio_offset_s)||0,scale=Number(a.bio_scale)||1,b=D.biosignals||{};
 const conv=t=>(t||[]).map(x=>x*scale+off);
 const ecg=signals.ecg||b.ecg||{};
 plot("signals",[
  {name:"ЭКГ, мВ",color:"#007c78",t:conv(ecg.t_s),y:ecg.mv||ecg.max_mv},
  {name:"rPPG, отн.",color:"#ce6536",t:D.video.times,y:D.video.rppg},
  {name:"RIP, мВ",color:"#74559c",t:conv(b.rip?.t_s),y:b.rip?.respiration_mv},
  {name:"GSR, мкСм",color:"#9d7b2b",t:conv(b.gsr?.t_s),y:b.gsr?.us||b.gsr?.microsiemens}
 ],current,20);
}
function drawSpectrum(){
 const c=$("spectrum"),w=c.clientWidth||400,h=180,dpr=devicePixelRatio||1;c.width=w*dpr;c.height=h*dpr;
 c.style.height=`${h}px`;
 const ctx=c.getContext("2d");ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
 const bands=[[.0033,.04,"#eee9e0","VLF"],[.04,.15,"#e2efe7","LF"],[.15,.4,"#e3eaf3","HF"]];
 bands.forEach(([a,b,color,label])=>{ctx.fillStyle=color;ctx.fillRect(a/.5*w,0,(b-a)/.5*w,h-24);ctx.fillStyle="#607071";ctx.font="10px Segoe UI";ctx.fillText(label,(a+b)/2/.5*w-10,14)});
 const s=signals.spectrum;if(!s){ctx.fillStyle="#607071";ctx.fillText("Недостаточно валидных RR интервалов",30,90);return}
 const max=Math.max(...s.power,1);ctx.strokeStyle="#007c78";ctx.lineWidth=1.5;ctx.beginPath();s.f.forEach((f,i)=>{const x=f/.5*w,y=h-25-s.power[i]/max*(h-48);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y)});ctx.stroke();
 ctx.fillStyle="#607071";ctx.fillText("0",0,h-4);ctx.fillText("0.5 Гц",w-40,h-4);
}
function drawOverlay(){
 const c=$("overlay"),w=c.clientWidth,h=c.clientHeight;c.width=w;c.height=h;const ctx=c.getContext("2d");
 if(!$("overlayToggle").checked||!D.video.overlays?.length)return;
 const o=near(D.video.overlays,"t",current);if(Math.abs(o.t-current)>.4)return;
 for(const [name,pts] of Object.entries(o.rois||{})){
  ctx.strokeStyle=name==="mouth"||name==="eyes"?"#f7ca78":"#68f0c0";ctx.lineWidth=1;ctx.beginPath();
  pts.forEach(([x,y],i)=>i?ctx.lineTo(x*w,y*h):ctx.moveTo(x*w,y*h));ctx.closePath();ctx.stroke();
 }
 ctx.strokeStyle="#80bfff";ctx.lineWidth=1.5;
 for(const [a,b] of D.video.pose_edges||[]){const p=o.pose[a],q=o.pose[b];if(p?.[2]&&q?.[2]){ctx.beginPath();ctx.moveTo(p[0]*w,p[1]*h);ctx.lineTo(q[0]*w,q[1]*h);ctx.stroke()}}
}
function dashboard(){
 const a=D.annotation,bt=(current-a.bio_offset_s)/(a.bio_scale||1),hrv=near(D.hrv,"center_s",bt)||{};
 const v=near(D.video.windows,"start_s",current-15)||{},ac=near(D.audio_features,"time_s",current)||{};
 const segs=D.audio.segments||[],ix=segs.findIndex(s=>current>=s.start&&current<=s.end),seg=segs[ix];
 let latency=null,words=null;
 if(seg){const speaker=seg.speaker||seg.speaker_id,prev=segs[ix-1];words=(seg.text.trim().split(/\s+/).length/Math.max(seg.end-seg.start,.01))*60;
  if(a.speaker_roles?.[speaker]==="respondent"&&prev&&a.speaker_roles?.[prev.speaker||prev.speaker_id]==="interviewer")latency=seg.start-prev.end;}
 const inRange=bt>=0&&bt<=(D.bio.analysis_duration_s||0);
 const hvalid=inRange&&hrv.ecg_quality==="usable_automatic";
 const metrics=[["ЧСС ЭКГ",hvalid?hrv.hr_bpm:null,"уд/мин"],["ЧСС rPPG",v.rppg_hr_bpm,"уд/мин"],["RMSSD",hvalid?hrv.rmssd_ms:null,"мс"],["Основной тон",ac.pitch_median_hz,"Гц"],["Темп сегмента",words,"слов/мин"],["Задержка ответа",latency,"с"]];
 $("metrics").replaceChildren(...metrics.map(([name,val,unit])=>{const div=node("div","","metric");div.append(node("span",name),node("strong",fmt(val)),node("span",unit));return div}));
 $("comparison").replaceChildren(node("p",`LF/HF: ${fmt(hvalid?hrv.lf_hf:null,2)} · Дыхание RIP: ${fmt(inRange?hrv.resp_bpm:null)} /мин`),
 node("p",`Частота дыхательного движения видео: ${fmt(v.respiratory_motion_bpm)} /мин · качество rPPG: ${fmt(v.spectral_concentration,2)}`),
 node("p",`Межмодальное сравнение: ${a.sync_verified?"синхронизация подтверждена пользователем":"ожидает проверки временного смещения"}`),
 node("p","Оценка правдивости: недоступна без проверенной экспериментальной разметки и независимой валидации."));
}
function update(t){
 current=t;$("clock").textContent=clock(t);$("time").value=t;
 $("syncState").textContent=D.annotation?.sync_verified?"Временное совмещение проверено":"Временное совмещение не проверено";
 const idx=(D.audio.segments||[]).findIndex(s=>t>=s.start&&t<s.end);
 [...$("transcript").children].forEach((b,i)=>b.classList.toggle("active",i===idx));
 drawSignals();drawOverlay();dashboard();
 if(D.annotation)fetchSignal((t-D.annotation.bio_offset_s)/(D.annotation.bio_scale||1));

}
$("sessions").onchange=()=>load($("sessions").value).catch(e=>notify(e.message));
$("reload").onclick=()=>load(sid).catch(e=>notify(e.message));
$("time").oninput=()=>seek(Number($("time").value));
$("prev").onclick=()=>{video.pause();seek(current-1/(D.video.source_fps||25))};
$("next").onclick=()=>{video.pause();seek(current+1/(D.video.source_fps||25))};
$("play").onclick=async()=>{if(!D.media_ready){notify("Видео ещё обрабатывается");return}if(video.paused){try{await video.play();}catch(e){notify(e.message)}}else{video.pause();}};
video.onplay=()=>$("play").textContent="Пауза";video.onpause=()=>$("play").textContent="Воспроизвести";
video.onloadedmetadata=()=>{$("time").max=video.duration;renderEvents()};
video.ontimeupdate=()=>{if(performance.now()-lastRefresh>80){lastRefresh=performance.now();update(video.currentTime)}};
video.onseeked=()=>update(video.currentTime);
$("speed").onchange=()=>{video.playbackRate=Number($("speed").value)};
$("storySelect").onchange=showStory;
$("saveSync").onclick=async()=>{D.annotation.bio_offset_s=Number($("offset").value);D.annotation.bio_scale=Number($("scale").value);D.annotation.sync_verified=$("verified").checked;await save();signalAt=-999;update(current)};
$("setStart").onclick=()=>$("eventStart").value=current.toFixed(2);$("setEnd").onclick=()=>$("eventEnd").value=current.toFixed(2);
$("addEvent").onclick=async()=>{const e={start:Number($("eventStart").value),end:Number($("eventEnd").value),label:$("eventLabel").value,note:$("eventNote").value,verified:$("eventVerified").checked,story_id:$("storySelect").value||null};if(e.end<=e.start){notify("Конец должен быть позже начала");return}D.annotation.events.push(e);await save();renderEvents()};

$("overlayToggle").onchange=drawOverlay;
window.addEventListener("resize",()=>{drawSignals();drawSpectrum();drawOverlay()});
init().catch(e=>{$("status").textContent=e.message});
