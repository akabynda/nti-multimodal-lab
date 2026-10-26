const $=id=>document.getElementById(id);
let D={},list=[],sid="",current=0,signals={},signalAt=-999,lastRefresh=0,generation=0,signalGeneration=0,loading=false,activeTranscriptIndex=-1,redrawQueued=false;
let analysisGeneration=0;
const video=$("video");
const labelText={truth:"Правда",lie:"Ложь",baseline:"База",question:"Вопрос",answer:"Ответ",topic:"Тема",unknown:"Неизвестно"};
const roleNames={interviewer:"Интервьюер",respondent:"Респондент",unknown:"Говорящий"};
const transcriptRows=()=>D.audio.turns||D.audio.segments||[];
const dialogueRole=s=>roleNames[s?.speaker_role]?s.speaker_role:(D.annotation?.speaker_roles?.[s?.speaker||s?.speaker_id]||"unknown");
const clock=t=>`${Math.floor(t/60).toString().padStart(2,"0")}:${(t%60).toFixed(2).padStart(5,"0")}`;
const fmt=(n,d=1)=>n!==null&&n!==undefined&&n!==""&&Number.isFinite(Number(n))?Number(n).toFixed(d):"—";
const near=(a,key,t)=>a?.length?a.reduce((best,r)=>Math.abs(Number(r[key])-t)<Math.abs(Number(best[key])-t)?r:best,a[0]):{};
function node(tag,text,cls){const e=document.createElement(tag);e.textContent=text;if(cls)e.className=cls;return e}
async function json(url,opts){const r=await fetch(url,opts);if(!r.ok)throw Error(`${r.status}: ${await r.text()}`);return r.json()}
function notify(s){$("notice").textContent=s}
function text(id,value){if($(id))$(id).textContent=value}
function setStatus(message,state){$("status").textContent=message;$("status").dataset.state=state}
function duration(t){const seconds=Math.max(0,Math.floor(Number(t)||0));return `${Math.floor(seconds/60).toString().padStart(2,"0")}:${(seconds%60).toString().padStart(2,"0")}`}
function setSessionControls(ready){
 document.querySelectorAll("button,input,select").forEach(e=>{
  const independent=e.id==="sessions"||e.id==="reload"||e.id==="followTranscript"||e.closest("#textTabs");
  e.disabled=!ready&&!independent;
 });
}
function clearSessionView(message){
 D={video:{},audio:{},bio:{},biosignals:{},annotation:null,stories:[],matches:[]};
 renderProfile();
 signals={};current=0;signalAt=-999;activeTranscriptIndex=-1;
 video.pause();video.removeAttribute("src");video.load();
 for(const id of ["storySelect","match","evidence","markers","events","roles","baselines","metrics","comparison"])$(id).replaceChildren();
 $("transcript").textContent=message;$("transcript").scrollTop=0;
 $("story").textContent=message;
 if($("referencePanel"))$("referencePanel").scrollTop=0;
 $("time").max=0;$("time").value=0;$("offset").value=0;$("scale").value=1;
 text("clock",clock(0));text("duration","00:00");text("transcriptCount","0");text("storyCount","0");
 text("asrInfo","—");text("faceCoverage","");text("play","Воспроизвести");
 $("ppgLegend").hidden=true;
 for(const id of ["signals","spectrum","overlay"]){const c=$(id);c.getContext("2d").clearRect(0,0,c.width,c.height)}
}
async function init(){
 setSessionControls(false);
 setStatus("Загрузка записей…","loading");
 list=await json("/api/sessions");
 $("sessions").replaceChildren(...list.map(s=>{let o=node("option",s.id.replace("_"," · "));o.value=s.id;return o}));

 if(list.length)await load(list[0].id);
 else {clearSessionView("Записи ещё не добавлены.");setStatus("Нет записей. Запустите обработку данных по README.","ready");setSessionControls(false)}
}
async function load(id){
 const requestGeneration=++generation;
 loading=true;++signalGeneration;sid=id;
 clearSessionView("Загрузка записи…");setSessionControls(false);notify("");
 $("sessions").value=id;
 const number=Number(id.split("_")[0])||list.findIndex(s=>s.id===id)+1;
 text("sessionNumber",`Запись ${String(number).padStart(2,"0")}`);
 setStatus("Загрузка записи…","loading");
 try{
  const response=await json(`/api/session/${encodeURIComponent(id)}`);
  if(requestGeneration!==generation)return;
  D=response;
  if(D.media_ready){video.src=`/media/${encodeURIComponent(sid)}`;video.load()}
  $("time").max=D.video.duration_s||D.audio.duration_s||D.bio.analysis_duration_s||1200;
  text("duration",duration($("time").max));
  $("offset").value=D.annotation.bio_offset_s||0;$("scale").value=D.annotation.bio_scale||1;
  const alignment=D.alignment||{};
  $("baselines").replaceChildren(...(alignment.baselines||[]).map(b=>{const button=node("button",`${b.name==="beginning"?"Начальный":"Конечный"} фон · ${clock(b.start_s)}–${clock(b.end_s)}`);button.onclick=()=>seek(b.start_s);return button}));
  const segmentCount=transcriptRows().length;
  setStatus(`Реплики: ${segmentCount} · каналы: ${D.bio.channels?.length||0}`,"ready");
  text("transcriptCount",String(segmentCount));text("storyCount",String(D.stories.length));
  $("asrInfo").textContent=D.audio.editorial_review?"Whisper large-v3 · текст вычитан":D.audio.model||"Нет расшифровки";
  $("faceCoverage").textContent=`Лицо отслежено: ${fmt((D.video.face_coverage??NaN)*100,1)}% кадров`;
  $("ppgLegend").hidden=!D.biosignals?.ppg?.available;
  renderProfile();renderTranscript();renderStories();renderEvents();renderRoles();
  loading=false;setSessionControls(true);update(0);
  if(!D.media_ready)notify("Видео ещё обрабатывается. Нажмите «Обновить данные» после обработки.");
 }catch(e){
  if(requestGeneration!==generation)return;
  clearSessionView("Запись не загружена. Нажмите «Обновить данные», чтобы повторить.");
  setStatus("Не удалось загрузить запись","error");
  notify(`Ошибка загрузки: ${e.message}. Повторите через «Обновить данные».`);
 }finally{
  if(requestGeneration===generation){loading=false;setSessionControls(!!D.annotation)}
 }
}
function renderProfile(){
 const panel=$("participantProfile"),fields=$("profileFields"),profile=D.profile;
 fields.replaceChildren();panel.open=false;panel.hidden=!profile;
 text("profileNotice","");$("profileNotice").hidden=true;
 text("profileSummary","Данные участника");
 if(!profile)return;
 const value=v=>v===null||v===undefined||v===""?"—":String(v);
 text("profileSummary",`Данные участника · ИСУ ${value(profile.tab_number)}`);
 const entries=[["Имя",D.profile_display_name,"profile-name"]];
 if(profile.profile_link_status==="requested_id_not_found"){
  text("profileNotice","В таблице нет строки с этим ИСУ.");$("profileNotice").hidden=false;
 }else entries.push(
  ["Естественный",profile.profile_natural_raw,"profile-wide"],
  ["Демонстрируемый",profile.profile_demonstrated_raw,"profile-wide"],
  ["Выраженность",profile.expression],
  ["Эмоц+",profile.emotion_positive],
  ["Эмоц−",profile.emotion_negative],
  ["Эмоц_Баланс",profile.emotion_balance]
 );
 for(const [label,raw,cls] of entries){
  const field=node("div","",cls);
  field.append(node("dt",label),node("dd",value(raw)));fields.append(field);
 }
}
async function loadAnalysis(){
 const requestGeneration=++analysisGeneration;
 $("multimodalStatus").hidden=false;$("multimodalResults").hidden=true;
 text("multimodalStatus","Загрузка анализа…");
 try{
  const result=await json("/api/analysis");
  if(requestGeneration!==analysisGeneration)return;
  renderAnalysis(result);
 }catch(e){
  if(requestGeneration!==analysisGeneration)return;
  text("multimodalSummary","Модели и корреляции");
  text("multimodalStatus","Не удалось загрузить анализ. Нажмите «Обновить данные», чтобы повторить.");
 }
}
function renderAnalysis(result){
 const ready=result&&result.status!=="not_run";
 $("multimodalStatus").hidden=!!ready;$("multimodalResults").hidden=!ready;
 text("multimodalSummary","Модели и корреляции");
 if(!ready){text("multimodalStatus","Анализ ещё не рассчитан.");return}
 const number=(value,d=3)=>fmt(value,d).replace(".",",");
 text("multimodalModelCaption",`Сравнение моделей · выборка из ${number(result.dataset?.trials,0)} ответов`);
 const models=(result.models||[]).map(model=>{
  const row=node("tr",""),name=node("th",model.name||model.id||"—");name.scope="row";row.append(name);
  for(const metric of ["brier","balanced_accuracy","roc_auc"])row.append(node("td",number(model.metrics?.[metric])));
  return row;
 });
 if(!models.length){const row=node("tr",""),cell=node("td","Модели ещё не рассчитаны.");cell.colSpan=4;row.append(cell);models.push(row)}
 $("multimodalModels").replaceChildren(...models);
 const increment=result.profile_increment||{};
 text("multimodalIncrement",`Δ Brier (с профилями − без профилей): ${number(increment.delta_brier)} · 95% интервал [${number(increment.ci_low)}; ${number(increment.ci_high)}]`);
 const associations=result.associations||[];
 $("multimodalAssociations").hidden=!associations.length;
 text("multimodalAssociationsSummary",`Связи профилей с показателями · ${associations.length}`);
 $("multimodalAssociationRows").replaceChildren(...associations.map(association=>{
  const row=node("tr",""),profile=node("th",association.profile_label||"—");profile.scope="row";
  row.append(profile,node("td",association.feature_label||"—"),node("td",number(association.n,0)),node("td",number(association.rho)),node("td",number(association.p_holm,4)));
  return row;
 }));
}
function renderTranscript(){
 const segs=transcriptRows();
 activeTranscriptIndex=-1;$("transcript").scrollTop=0;
 $("transcript").replaceChildren(...segs.map((s,i)=>{
  const b=node("button","");b.dataset.index=i;b.dataset.speaker=dialogueRole(s);
  const voice=node("span",roleNames[dialogueRole(s)],"speaker");
  b.append(node("span",clock(s.start),"time"),voice,node("span",s.text,"utterance-text"));
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
 $("match").replaceChildren();
 if(s?.source_url){const a=node("a","Открыть источник");a.href=s.source_url;a.target="_blank";a.rel="noopener";$("match").append(a)}
 $("evidence").replaceChildren(...(D.comparisons||[]).filter(x=>x.story_id===s?.story_id).map(renderFinding));
 if($("referencePanel"))$("referencePanel").scrollTop=0;
}
function evidenceField(parent,label,value,quoted=true){
 if(!value)return;
 const field=node("div","","evidence-field");
 field.append(node("span",label,"evidence-label"),node(quoted?"blockquote":"p",value));parent.append(field);
}
function renderFinding(r){
 const names={consistent:"Согласуется с эталоном",contradiction_candidate:"Несоответствие эталону",different:"Дополнительные сведения",omitted:"Эпизод не рассказан",uncertain:"Неоднозначный ответ"};
 const box=node("article","",`finding ${r.relation}`),head=node("div","","finding-head");
 head.append(node("strong",names[r.relation]||r.relation,"finding-status"));
 if(r.start_s!==null&&r.start_s!==undefined&&Number.isFinite(Number(r.start_s))){
  const start=r.context_start_s??r.start_s,b=node("button",`Слушать ответ ${clock(r.start_s)}`,"finding-play");
  b.setAttribute("aria-label",`Слушать вопрос и ответ с ${clock(start)}`);b.onclick=()=>playFrom(start);head.append(b);
 }
 box.append(head);
 if(r.question_summary)evidenceField(box,"Смысл вопроса",r.question_summary,false);
 else evidenceField(box,"Вопрос",r.question_context);
 evidenceField(box,"Эталон",r.source_excerpt||r.source_detail,!!r.source_excerpt);
 evidenceField(box,"Ответ",r.spoken_excerpt);
 for(const extra of r.additional_spoken_evidence||[]){
  const followup=node("div","","finding-followup"),b=node("button",`Слушать реплику ${clock(extra.start_s)}`,"finding-play");
  b.onclick=()=>playFrom(extra.start_s);followup.append(b);
  evidenceField(followup,"Следующий вопрос",extra.question_context);evidenceField(followup,"Ответ",extra.spoken_excerpt);box.append(followup);
 }
 if(r.context_excerpt){const details=node("details","");details.append(node("summary","Контекст вопроса и ответа"),node("p",r.context_excerpt));box.append(details)}
 return box;
}
function renderRoles(){
 if(D.audio.editorial_review){
  const rows=transcriptRows();
  $("roles").replaceChildren(...Object.entries(roleNames).filter(([role])=>rows.some(s=>dialogueRole(s)===role)).map(([role,label])=>node("span",`${label}: ${rows.filter(s=>dialogueRole(s)===role).length}`)));
  return;
 }
 const speakers=[...new Set((D.audio.segments||[]).map(s=>s.speaker||s.speaker_id).filter(Boolean))];
 $("roles").replaceChildren(...speakers.map(s=>{
  const l=node("label",s),sel=document.createElement("select");
  [["unknown","Не назначена"],["interviewer","Интервьюер"],["respondent","Респондент"]].forEach(([v,t])=>{const o=node("option",t);o.value=v;sel.append(o)});
  sel.value=D.annotation.speaker_roles?.[s]||"unknown";
  sel.onchange=async()=>{D.annotation.speaker_roles??={};D.annotation.speaker_roles[s]=sel.value;await save();renderTranscript();update(current)};
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
async function save(){const savedGeneration=generation;await json(`/api/session/${sid}/annotation`,{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify(D.annotation)});if(savedGeneration===generation)notify("Разметка сохранена локально")}
function seek(t){if(loading||!D.annotation||!Number.isFinite(Number(t)))return;current=Math.max(0,Math.min(Number($("time").max),Number(t)));if(D.media_ready)video.currentTime=current;update(current)}
async function playFrom(t){
 if(loading||!D.annotation)return;
 if(!D.media_ready){notify("Видео ещё обрабатывается. Нажмите «Обновить данные» после обработки.");return}
 const playedGeneration=generation;
 try{seek(t);await video.play()}catch(e){if(playedGeneration===generation)notify(`Не удалось начать воспроизведение: ${e.message}`)}
}
function followTranscript(){
 if(!$("followTranscript")?.checked||$("transcriptPanel")?.hidden||activeTranscriptIndex<0)return;
 const container=$("transcript"),row=container.children[activeTranscriptIndex];
 if(!row)return;
 const viewport=container.getBoundingClientRect(),item=row.getBoundingClientRect(),padding=8;
 if(item.top<viewport.top+padding||item.height>viewport.height)container.scrollTop+=item.top-viewport.top-padding;
 else if(item.bottom>viewport.bottom-padding)container.scrollTop+=item.bottom-viewport.bottom+padding;
}
function requestCanvasRedraw(){
 if(redrawQueued)return;
 redrawQueued=true;
 requestAnimationFrame(()=>{redrawQueued=false;if(!loading&&D.annotation){drawSignals();drawSpectrum();drawOverlay()}});
}
function activateTextTab(tab,focus=false){
 const tabs=[...$("textTabs").querySelectorAll('[role="tab"]')];
 for(const item of tabs){
  const active=item===tab,panel=$(item.dataset.panel);
  item.setAttribute("aria-selected",String(active));item.tabIndex=active?0:-1;
  item.setAttribute("aria-controls",item.dataset.panel);
  if(panel){panel.hidden=!active;panel.setAttribute("role","tabpanel");panel.setAttribute("aria-labelledby",item.id)}
 }
 if(focus)tab.focus();
 if(tab.dataset.panel==="transcriptPanel")followTranscript();
 requestCanvasRedraw();
}
function initTextTabs(){
 const tablist=$("textTabs");if(!tablist)return;
 tablist.setAttribute("role","tablist");
 const tabs=[...tablist.querySelectorAll('[role="tab"]')];
 for(const [i,tab] of tabs.entries()){
  tab.onclick=()=>activateTextTab(tab);
  tab.onkeydown=e=>{
   const vertical=tablist.getAttribute("aria-orientation")==="vertical";
   let next=i;
   if(e.key===(vertical?"ArrowDown":"ArrowRight"))next=(i+1)%tabs.length;
   else if(e.key===(vertical?"ArrowUp":"ArrowLeft"))next=(i+tabs.length-1)%tabs.length;
   else if(e.key==="Home")next=0;
   else if(e.key==="End")next=tabs.length-1;
   else return;
   e.preventDefault();activateTextTab(tabs[next],true);
  };
 }
 if(tabs.length)activateTextTab(tabs.find(t=>t.getAttribute("aria-selected")==="true")||tabs[0]);
}
function plot(id,traces,t,span){
 const c=$(id),w=c.clientWidth||800,h=Number(c.dataset.logicalHeight||240),dpr=devicePixelRatio||1;
 c.style.height=`${h}px`;
 c.width=w*dpr;c.height=h*dpr;const ctx=c.getContext("2d");ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
 const chartHeight=h-24,row=chartHeight/Math.max(traces.length,1),left=88,right=w-8,range=span||20,x0=t-range/2;
 ctx.font="11px Segoe UI";
 for(let tick=0;tick<=4;tick++){
  const x=left+(right-left)*tick/4,value=x0+range*tick/4;
  ctx.strokeStyle=tick===2?"#93a5cc":"#edf0f6";ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,chartHeight);ctx.stroke();
  ctx.fillStyle="#5b5e66";ctx.textAlign=tick===0?"left":tick===4?"right":"center";
  ctx.fillText(`${value<0?"−":""}${duration(Math.abs(value))}`,x,h-5);
 }
 ctx.textAlign="left";
 traces.forEach((s,i)=>{
  const top=i*row;ctx.strokeStyle="#edf0f6";ctx.beginPath();ctx.moveTo(left,top+row/2);ctx.lineTo(right,top+row/2);ctx.stroke();
  ctx.font="11px Segoe UI";ctx.fillStyle="#5b5e66";ctx.fillText(s.name,0,top+row/2-3);
  ctx.fillStyle=s.color;ctx.fillRect(0,top+row/2+4,16,2);
  let pairs=(s.t||[]).map((x,j)=>[Number(x),s.y?.[j]]).filter(p=>p[0]>=x0&&p[0]<=x0+range);
  const vals=pairs.map(x=>x[1]).filter(x=>x!==null&&Number.isFinite(Number(x))).map(Number);
  if(!vals.length){ctx.fillStyle="#5b5e66";ctx.fillText("нет валидных данных",left+8,top+row/2);return}
  let lo=Math.min(...vals),hi=Math.max(...vals),amp=Math.max(hi-lo,1e-8);
  ctx.strokeStyle=s.color;ctx.lineWidth=1.1;ctx.beginPath();let pen=false;
  pairs.forEach(([x,y])=>{if(y===null||!Number.isFinite(Number(y))){pen=false;return}const px=left+(x-x0)/range*(right-left),py=top+row*.85-(y-lo)/amp*row*.7;if(pen)ctx.lineTo(px,py);else{ctx.moveTo(px,py);pen=true}});ctx.stroke();
 });
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
 const traces=[
  {name:"ЭКГ, мВ",color:"#0B68FE",t:conv(ecg.t_s),y:ecg.mv||ecg.max_mv},
  {name:"rPPG, отн.",color:"#F93F37",t:D.video.times,y:D.video.rppg},
  {name:"RIP, мВ",color:"#0084A6",t:conv(b.rip?.t_s),y:b.rip?.respiration_mv},
  {name:"GSR, мкСм",color:"#5B5E66",t:conv(b.gsr?.t_s),y:b.gsr?.us||b.gsr?.microsiemens}
 ];
 if(b.ppg?.available)traces.splice(2,0,{name:"ФПГ, отн.",color:"#7F00FF",t:conv(b.ppg.t_s),y:b.ppg.filtered});
 $("signals").dataset.logicalHeight=b.ppg?.available?300:240;
 plot("signals",traces,current,20);
}
function drawSpectrum(){
 const c=$("spectrum"),w=c.clientWidth||400,h=180,dpr=devicePixelRatio||1;c.width=w*dpr;c.height=h*dpr;
 c.style.height=`${h}px`;
 const ctx=c.getContext("2d");ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
 const bands=[[.0033,.04,"#f3ebff","VLF"],[.04,.15,"#edf4ff","LF"],[.15,.4,"#e5f8fd","HF"]];
 bands.forEach(([a,b,color,label])=>{ctx.fillStyle=color;ctx.fillRect(a/.5*w,0,(b-a)/.5*w,h-24);ctx.fillStyle="#5b5e66";ctx.font="10px Segoe UI";ctx.fillText(label,(a+b)/2/.5*w-10,14)});
 const s=signals.spectrum;if(!s){ctx.fillStyle="#5b5e66";ctx.fillText("Недостаточно валидных RR интервалов",30,90);return}
 const max=Math.max(...s.power,1);ctx.strokeStyle="#0B68FE";ctx.lineWidth=1.5;ctx.beginPath();s.f.forEach((f,i)=>{const x=f/.5*w,y=h-25-s.power[i]/max*(h-48);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y)});ctx.stroke();
 ctx.fillStyle="#5b5e66";ctx.fillText("0",0,h-4);ctx.fillText("0.5 Гц",w-40,h-4);
}
function drawOverlay(){
 const c=$("overlay"),w=c.clientWidth,h=c.clientHeight;c.width=w;c.height=h;const ctx=c.getContext("2d");
 if(!$("overlayToggle").checked||!D.video.overlays?.length)return;
 const o=near(D.video.overlays,"t",current);if(Math.abs(o.t-current)>.4)return;
 for(const [name,pts] of Object.entries(o.rois||{})){
  ctx.strokeStyle=name==="mouth"||name==="eyes"?"#D0FF14":"#00CCFF";ctx.lineWidth=1;ctx.beginPath();
  pts.forEach(([x,y],i)=>i?ctx.lineTo(x*w,y*h):ctx.moveTo(x*w,y*h));ctx.closePath();ctx.stroke();
 }
 ctx.strokeStyle="#ffffff";ctx.lineWidth=1.5;
 for(const [a,b] of D.video.pose_edges||[]){const p=o.pose[a],q=o.pose[b];if(p?.[2]&&q?.[2]){ctx.beginPath();ctx.moveTo(p[0]*w,p[1]*h);ctx.lineTo(q[0]*w,q[1]*h);ctx.stroke()}}
}
function dashboard(){
 const a=D.annotation,bt=(current-a.bio_offset_s)/(a.bio_scale||1),hrv=near(D.hrv,"center_s",bt)||{};
 const v=near(D.video.windows,"start_s",current-15)||{},ac=near(D.audio_features,"time_s",current)||{};
 const segs=transcriptRows(),ix=segs.findIndex(s=>current>=s.start&&current<=s.end),seg=segs[ix];
 let latency=null,words=null;
 if(seg){words=(seg.text.trim().split(/\s+/).length/Math.max(seg.end-seg.start,.01))*60;
  const pair=(D.audio.qa_candidates||[]).find(p=>p.method==="editorially_reviewed_dialogue_roles"&&current>=p.answer_start_s&&current<=p.answer_end_s);
  if(pair&&pair.latency_s>=0)latency=pair.latency_s;
  else if(!D.audio.editorial_review){const prev=segs[ix-1];if(dialogueRole(seg)==="respondent"&&prev&&dialogueRole(prev)==="interviewer"&&seg.start>=prev.end)latency=seg.start-prev.end;}}
 const inRange=bt>=0&&bt<=(D.bio.analysis_duration_s||0);
 const hvalid=inRange&&hrv.ecg_quality==="usable_automatic";
 const metrics=[["ЧСС ЭКГ",hvalid?hrv.hr_bpm:null,"уд/мин"],["ЧСС rPPG",v.rppg_hr_bpm,"уд/мин"],["RMSSD",hvalid?hrv.rmssd_ms:null,"мс"],["Индекс напряжения",hvalid?hrv.stress_index:null,"у. е."],["Основной тон",ac.pitch_median_hz,"Гц"],["Темп сегмента",words,"слов/мин"],["Задержка ответа",latency,"с"]];
 $("metrics").replaceChildren(...metrics.map(([name,val,unit])=>{const div=node("div","","metric");div.append(node("span",name),node("strong",fmt(val)),node("span",unit));return div}));
 $("comparison").replaceChildren(node("p",`LF/HF: ${fmt(hvalid?hrv.lf_hf:null,2)} · Дыхание RIP: ${fmt(inRange?hrv.resp_bpm:null)} /мин`),
 node("p",`Частота дыхательного движения видео: ${fmt(v.respiratory_motion_bpm)} /мин · качество rPPG: ${fmt(v.spectral_concentration,2)}`));
 if(D.biosignals?.ppg?.available){
  const pw=near(D.ppg_metrics,"center_s",bt)||{},pv=inRange&&pw.ppg_quality==="usable_automatic";
  $("comparison").prepend(node("p",`Контактный ФПГ: ${fmt(pv?pw.hr_bpm:null)} уд/мин · RMSSD: ${fmt(pv?pw.rmssd_ms:null)} мс`));
 }
}
function update(t){
 if(loading||!D.annotation)return;
 current=t;$("clock").textContent=clock(t);$("time").value=t;
 const idx=transcriptRows().findIndex(s=>t>=s.start&&t<s.end);
 if(idx!==activeTranscriptIndex){
  const previous=$("transcript").children[activeTranscriptIndex],active=$("transcript").children[idx];
  if(previous){previous.classList.remove("active");previous.removeAttribute("aria-current")}
  if(active){active.classList.add("active");active.setAttribute("aria-current","true")}
  activeTranscriptIndex=idx;followTranscript();
 }
 drawSignals();drawOverlay();dashboard();
 if(D.annotation)fetchSignal((t-D.annotation.bio_offset_s)/(D.annotation.bio_scale||1));

}
$("sessions").onchange=()=>load($("sessions").value).catch(e=>notify(e.message));
$("reload").onclick=()=>{loadAnalysis();const action=sid?load(sid):init();action.catch(e=>{setStatus("Не удалось загрузить список записей","error");notify(`${e.message}. Нажмите «Обновить данные», чтобы повторить.`)})};
$("time").oninput=()=>seek(Number($("time").value));
$("prev").onclick=()=>{video.pause();seek(current-1/(D.video.source_fps||25))};
$("next").onclick=()=>{video.pause();seek(current+1/(D.video.source_fps||25))};
$("play").onclick=async()=>{if(!D.media_ready){notify("Видео ещё обрабатывается");return}if(video.paused){try{await video.play();}catch(e){notify(e.message)}}else{video.pause();}};
video.onplay=()=>$("play").textContent="Пауза";video.onpause=()=>$("play").textContent="Воспроизвести";
video.onloadedmetadata=()=>{if(loading||!D.media_ready)return;if(Number.isFinite(video.duration)){$("time").max=video.duration;text("duration",duration(video.duration))}video.playbackRate=Number($("speed").value);renderEvents();requestCanvasRedraw()};
video.ontimeupdate=()=>{if(performance.now()-lastRefresh>80){lastRefresh=performance.now();update(video.currentTime)}};
video.onseeked=()=>update(video.currentTime);
$("speed").onchange=()=>{video.playbackRate=Number($("speed").value)};
$("storySelect").onchange=showStory;
$("saveSync").onclick=async()=>{
 const offset=Number($("offset").value),scale=Number($("scale").value);
 if(offset!==D.annotation.bio_offset_s||scale!==D.annotation.bio_scale)D.annotation.sync_verified=false;
 D.annotation.bio_offset_s=offset;D.annotation.bio_scale=scale;await save();signalAt=-999;update(current);
};
$("setStart").onclick=()=>$("eventStart").value=current.toFixed(2);$("setEnd").onclick=()=>$("eventEnd").value=current.toFixed(2);
$("addEvent").onclick=async()=>{const e={start:Number($("eventStart").value),end:Number($("eventEnd").value),label:$("eventLabel").value,note:$("eventNote").value,verified:$("eventVerified").checked,story_id:$("storySelect").value||null};if(e.end<=e.start){notify("Конец должен быть позже начала");return}D.annotation.events.push(e);await save();renderEvents()};

$("overlayToggle").onchange=drawOverlay;
if($("followTranscript"))$("followTranscript").onchange=followTranscript;
window.addEventListener("resize",requestCanvasRedraw);
document.querySelectorAll("details").forEach(details=>details.addEventListener("toggle",()=>{if(details.open)requestCanvasRedraw()}));
if(typeof ResizeObserver!=="undefined"){
 const observer=new ResizeObserver(requestCanvasRedraw);
 for(const element of [$("signals"),$("spectrum"),$("overlay").parentElement])if(element)observer.observe(element);
}
initTextTabs();
const compactLayout=window.matchMedia("(max-width: 800px)");
function positionTransport(){
 const workspace=document.querySelector(".review-workspace"),timeline=document.querySelector(".timeline");
 if(compactLayout.matches)workspace.before(timeline);else workspace.after(timeline);
 requestCanvasRedraw();
}
compactLayout.addEventListener("change",positionTransport);positionTransport();
loadAnalysis();
init().catch(e=>{setStatus("Не удалось загрузить список записей","error");notify(`${e.message}. Нажмите «Обновить данные», чтобы повторить.`);setSessionControls(false)});
