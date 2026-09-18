'use strict';

const $ = id => document.getElementById(id);
const fmt = value => Number(value).toLocaleString('en-US');
const STAGES = [
  {label:'ALERT',title:'Alert received',description:'Telemetry is loaded as inert data. No command, URL, path or payload is executed.'},
  {label:'PN',title:'PN representation',description:'Available categorical, text and numeric fields become a sparse engineered feature vector.'},
  {label:'KC',title:'KC expansion',description:'Sparse random connectivity expands the feature vector into Kenyon Cell-inspired activations.'},
  {label:'TOP-K',title:'Sparse fingerprint',description:'Winner-Take-All retains the strongest positive KC activations as a binary fingerprint.'},
  {label:'MEMORY',title:'Associative memory',description:'The fingerprint retrieves historical alerts and produces a nearest-neighbor novelty score.'},
  {label:'HYPOTHESIS',title:'Decision hypothesis',description:'A separate downstream classifier estimates an analyst verdict for review.'}
];
const MAX_FILE_BYTES = 5 * 1024 * 1024;
const MAX_RECORDS = 1000;
const state = {source:'manual',imported:[],importName:'',trace:null,stage:0,playing:false,timer:null,count:0,kcSample:[],edgeSample:[],reducedMotion:matchMedia('(prefers-reduced-motion: reduce)').matches};

function showError(message){$('error').textContent=message;$('error').hidden=false;clearTimeout(showError.timer);showError.timer=setTimeout(()=>$('error').hidden=true,9000);}
async function api(path,body){const response=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await response.json();if(!response.ok)throw new Error(data.error||`HTTP ${response.status}`);return data;}
function readable(value){return String(value||'Unknown').replaceAll('_',' ').replace(/\b\w/g,character=>character.toUpperCase());}
function setSource(source){state.source=source;for(const button of document.querySelectorAll('.source-tabs [data-source]'))button.setAttribute('aria-selected',String(button.dataset.source===source));for(const name of ['manual','json','file'])$(`source-${name}`).hidden=name!==source;$('input-status').textContent=source==='file'&&!state.imported.length?'Import a file to continue':'Ready to analyze';}

function normalizeRecord(record,index=0){
  if(!record||typeof record!=='object'||Array.isArray(record))throw new Error(`Alert ${index+1} must be a JSON object.`);
  const entries=Object.entries(record);
  if(!entries.length)throw new Error(`Alert ${index+1} is empty.`);
  if(entries.length>100)throw new Error(`Alert ${index+1} has more than 100 fields.`);
  const normalized=Object.create(null);
  for(const [rawKey,value] of entries){
    const key=String(rawKey).trim();
    if(!key)continue;
    if(value!==null&&typeof value==='object')throw new Error(`Alert ${index+1} field “${key}” is nested. Only flat records are supported.`);
    if(value===null||value===undefined||String(value).trim()==='')continue;
    const text=String(value);
    if(text.length>12000)throw new Error(`Alert ${index+1} field “${key}” is too long.`);
    normalized[key]=value;
  }
  if(!Object.keys(normalized).length)throw new Error(`Alert ${index+1} has no usable fields.`);
  if(JSON.stringify(normalized).length>60000)throw new Error(`Alert ${index+1} exceeds the local analysis limit.`);
  return normalized;
}

function parseJsonRecords(text){
  let value;
  try{value=JSON.parse(text.replace(/^\uFEFF/,''));}catch(error){throw new Error(`Invalid JSON: ${error.message}`);}
  const records=Array.isArray(value)?value:Array.isArray(value?.alerts)?value.alerts:[value];
  if(!records.length)throw new Error('The JSON file contains no alerts.');
  if(records.length>MAX_RECORDS)throw new Error(`The file contains more than ${fmt(MAX_RECORDS)} alerts.`);
  return records.map(normalizeRecord);
}

function parseDelimited(text,delimiter){
  const rows=[];let row=[],cell='',quoted=false;
  const input=text.replace(/^\uFEFF/,'');
  for(let i=0;i<input.length;i++){
    const character=input[i];
    if(quoted){if(character==='"'&&input[i+1]==='"'){cell+='"';i++;}else if(character==='"')quoted=false;else cell+=character;continue;}
    if(character==='"'){quoted=true;continue;}
    if(character===delimiter){row.push(cell);cell='';continue;}
    if(character==='\n'||character==='\r'){
      if(character==='\r'&&input[i+1]==='\n')i++;
      row.push(cell);cell='';if(row.some(value=>value.trim()))rows.push(row);row=[];continue;
    }
    cell+=character;
  }
  row.push(cell);if(row.some(value=>value.trim()))rows.push(row);
  if(quoted)throw new Error('The imported file contains an unclosed quoted value.');
  if(rows.length<2)throw new Error('The delimited file needs one header row and at least one alert.');
  const headers=rows[0].map(value=>value.trim());
  if(headers.some(value=>!value))throw new Error('Every imported column must have a header.');
  if(new Set(headers).size!==headers.length)throw new Error('Imported column headers must be unique.');
  const records=rows.slice(1).map(values=>Object.fromEntries(headers.map((header,index)=>[header,values[index]??''])));
  if(records.length>MAX_RECORDS)throw new Error(`The file contains more than ${fmt(MAX_RECORDS)} alerts.`);
  return records.map(normalizeRecord);
}

function parseFileText(file,text){
  const extension=file.name.toLowerCase().split('.').pop();
  if(extension==='json')return parseJsonRecords(text);
  if(extension==='tsv')return parseDelimited(text,'\t');
  if(extension==='csv')return parseDelimited(text,',');
  try{return parseJsonRecords(text);}catch(jsonError){try{return parseDelimited(text,text.includes('\t')?'\t':',');}catch{return (()=>{throw jsonError;})();}}
}

function setImported(records,name){
  state.imported=records;state.importName=name;
  $('import-name').textContent=name;$('import-count').textContent=`${fmt(records.length)} alert${records.length===1?'':'s'} ready`;
  $('import-row').replaceChildren(...records.map((record,index)=>{const option=document.createElement('option');option.value=String(index);option.textContent=`${String(index+1).padStart(3,'0')} · ${record.alert_id||record.alert_name||'Untitled alert'}`;return option;}));
  $('import-summary').hidden=false;$('import-preview').hidden=false;renderImportPreview(0);$('input-status').textContent=`Imported ${fmt(records.length)} alert${records.length===1?'':'s'}`;
}

function renderImportPreview(index){
  const record=state.imported[index];if(!record)return;
  const columns=Object.keys(record).slice(0,8),table=document.createElement('table'),head=document.createElement('thead'),headRow=document.createElement('tr'),body=document.createElement('tbody'),bodyRow=document.createElement('tr');
  for(const column of columns){const th=document.createElement('th');th.textContent=column;headRow.append(th);const td=document.createElement('td');td.textContent=String(record[column]??'');td.title=td.textContent;bodyRow.append(td);}
  head.append(headRow);body.append(bodyRow);table.append(head,body);$('import-preview').replaceChildren(table);
}

async function importFile(file){
  if(!file)return;
  if(file.size>MAX_FILE_BYTES)throw new Error('The selected file is larger than 5 MB.');
  $('input-status').textContent='Reading local file…';
  const records=parseFileText(file,await file.text());setImported(records,file.name);
}

function collectManual(){
  const record={};for(const input of document.querySelectorAll('[data-field]')){const value=input.value.trim();if(value)record[input.dataset.field]=value;}
  return normalizeRecord(record);
}
function collectJson(){const records=parseJsonRecords($('json-alert').value);if(records.length>1){setImported(records,'Pasted JSON');$('input-status').textContent=`Parsed ${fmt(records.length)} alerts · analyzing the first`;}$('json-alert').value=JSON.stringify(records[0],null,2);return records[0];}
function selectedAlert(){if(state.source==='manual')return collectManual();if(state.source==='json')return collectJson();if(!state.imported.length)throw new Error('Import a JSON, CSV or TSV file first.');return state.imported[Number($('import-row').value)||0];}

function probabilityRows(prediction){
  if(!prediction)return [];
  return Object.entries(prediction.probabilities||{}).sort((a,b)=>b[1]-a[1]).map(([label,value])=>{const row=document.createElement('div');row.className='probability-row';const name=document.createElement('span');name.textContent=readable(label);const track=document.createElement('div');track.className='probability-track';const fill=document.createElement('i');fill.style.width=`${Math.max(0,Math.min(100,Number(value)*100))}%`;track.append(fill);const score=document.createElement('strong');score.textContent=`${(Number(value)*100).toFixed(1)}%`;row.append(name,track,score);return row;});
}

function populateOutput(trace){
  const alert=trace.alert,prediction=trace.prediction;
  $('analysis-output').hidden=false;$('analysis-controls').hidden=false;
  $('output-alert-id').textContent=alert.alert_id||'MANUAL';$('output-alert-name').textContent=alert.alert_name||alert.process_name||'Untitled alert';
  $('output-alert-context').textContent=[alert.severity,alert.category,alert.hostname,alert.username,alert.mitre_technique].filter(Boolean).join(' · ')||'No additional context provided';
  $('output-pattern').textContent=trace.novelty.is_novel?'Unfamiliar pattern':'Recurrent pattern';$('output-pattern').className=`badge${trace.novelty.is_novel?' novel':''}`;
  $('output-pn').textContent=fmt(trace.pn_indices.length);$('output-kc').textContent=fmt(trace.active_kcs);$('output-novelty').textContent=Number(trace.novelty.novelty_score).toFixed(3);$('output-threshold').textContent=`threshold ${Number(trace.novelty_threshold).toFixed(3)}`;$('output-time').textContent=`${Number(trace.timing_ms.total).toFixed(1)} ms`;
  $('output-fingerprint-summary').textContent=`${fmt(trace.active_kcs)} / ${fmt(state.status.kc_dim)} ACTIVE`;
  $('output-fingerprint').replaceChildren(...trace.winner_indices.map(index=>{const dot=document.createElement('span');dot.className='fingerprint-dot';dot.title=`Active KC ${index}`;dot.setAttribute('aria-label',`Active KC ${index}`);return dot;}));
  if(prediction){$('output-hypothesis').textContent=readable(prediction.hypothesis||prediction.verdict);$('output-confidence').textContent=`${(Number(prediction.confidence)*100).toFixed(1)}%`;$('output-model').textContent=`Model: ${prediction.model} · separate from the FlySOC representation`;$('output-probabilities').replaceChildren(...probabilityRows(prediction));}
  else{$('output-hypothesis').textContent='Unavailable';$('output-confidence').textContent='—';$('output-model').textContent='No downstream classifier is available in this saved experiment.';$('output-probabilities').replaceChildren();}
  $('output-matches').replaceChildren(...trace.history.map((match,index)=>{const card=document.createElement('article');card.className='memory-match';const score=document.createElement('span');score.textContent=`${(Number(match.similarity)*100).toFixed(1)}%`;const name=document.createElement('strong');name.textContent=match.alert_name||'Untitled historical alert';name.title=name.textContent;const detail=document.createElement('small');detail.textContent=`${match.alert_id||'NO ID'} · ${readable(match.verdict||'UNLABELED')}`;card.append(score,name,detail);return card;}));
  const compactTrace={source:trace.source,alert:trace.alert,representation:{pn_nonzero:trace.pn_indices.length,kc_positive:trace.nonzero_activations,active_top_k:trace.active_kcs,winner_indices:trace.winner_indices},novelty:trace.novelty,nearest_alerts:trace.history,prediction:trace.prediction,timing_ms:trace.timing_ms};
  $('output-trace').textContent=JSON.stringify(compactTrace,null,2);
  state.kcSample=trace.kc_activations.map((value,index)=>({value:Number(value),index})).filter(item=>item.value>0).sort((a,b)=>b.value-a.value).slice(0,320);
  state.edgeSample=[];for(let i=0;i<Math.min(trace.contributing_edges.length,500);i+=2)state.edgeSample.push([trace.contributing_edges[i],trace.contributing_edges[i+1]]);
}

function renderStage(index){
  if(!state.trace)return;state.stage=Math.max(0,Math.min(STAGES.length-1,index));const stage=STAGES[state.stage];
  $('brain-stage-number').textContent=`STEP ${state.stage+1} / ${STAGES.length}`;$('brain-stage-title').textContent=stage.title;$('brain-stage-description').textContent=stage.description;$('analysis-position').textContent=`${String(state.stage+1).padStart(2,'0')} / ${String(STAGES.length).padStart(2,'0')}`;
  $('analysis-prev').disabled=state.stage===0;$('analysis-next').disabled=state.stage===STAGES.length-1;$('analysis-replay').textContent=state.playing?'Ⅱ Pause':'▶ Replay';
  $('analysis-stages').replaceChildren(...STAGES.map((item,itemIndex)=>{const button=document.createElement('button');button.type='button';button.textContent=item.label;button.className=`${itemIndex<state.stage?'complete ':''}${itemIndex===state.stage?'selected':''}`;button.addEventListener('click',()=>{stopReplay();renderStage(itemIndex);});return button;}));
  $('analyzer-fingerprint').classList.toggle('pending',state.stage<3);$('analyzer-memory').classList.toggle('pending',state.stage<4);$('analyzer-decision').classList.toggle('pending',state.stage<5);
  $('analysis-state').textContent=stage.label;
}
function stopReplay(){clearTimeout(state.timer);state.playing=false;if(state.trace)$('analysis-replay').textContent='▶ Replay';}
function queueReplay(){clearTimeout(state.timer);state.timer=setTimeout(()=>{if(!state.playing)return;if(state.stage>=STAGES.length-1){stopReplay();$('analysis-state').textContent='COMPLETE';return;}renderStage(state.stage+1);queueReplay();},620);}
function startReplay(){stopReplay();state.playing=true;renderStage(0);queueReplay();}

async function analyzeAlert(){
  if($('analyze-alert').disabled)return;
  try{
    const alert=selectedAlert();$('analyze-alert').disabled=true;$('input-status').textContent='Computing PN, KC, memory and hypothesis…';$('analysis-state').textContent='COMPUTING';
    const trace=await api('/api/fly/trace',{alert});state.trace=trace;populateOutput(trace);startReplay();state.count++;$('analysis-count').textContent=String(state.count);$('input-status').textContent=`Analyzed ${trace.alert.alert_id||'manual alert'} in ${Number(trace.timing_ms.total).toFixed(1)} ms`;
  }catch(error){showError(error.message);$('analysis-state').textContent='INPUT ERROR';$('input-status').textContent='Analysis could not be completed';}
  finally{$('analyze-alert').disabled=false;}
}

const canvas=$('brain-canvas'),context=canvas.getContext('2d');let canvasWidth=0,canvasHeight=0;
function resizeCanvas(){const rect=canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,2);canvas.width=Math.max(1,Math.round(rect.width*ratio));canvas.height=Math.max(1,Math.round(rect.height*ratio));context.setTransform(ratio,0,0,ratio,0,0);canvasWidth=rect.width;canvasHeight=rect.height;}
new ResizeObserver(resizeCanvas).observe(canvas);
function fraction(value){return Math.abs(Math.sin(value*12.9898)*43758.5453)%1;}
function pnPoint(index){return {x:canvasWidth*(.18+.64*fraction(index+5)),y:canvasHeight*(.72+.13*fraction(index+17))};}
function kcPoint(index){const side=index%2?-1:1,angle=6.283*fraction(index+29),radius=Math.sqrt(fraction(index+71));return {x:canvasWidth*(.5+side*(.09+.27*radius*Math.abs(Math.cos(angle)))),y:canvasHeight*(.38+.25*radius*Math.sin(angle))};}
function node(point,radius,color,alpha=1){context.globalAlpha=alpha;context.fillStyle=color;context.beginPath();context.arc(point.x,point.y,radius,0,Math.PI*2);context.fill();context.globalAlpha=1;}
function line(from,to,color,alpha=.15){context.globalAlpha=alpha;context.strokeStyle=color;context.lineWidth=.6;context.beginPath();context.moveTo(from.x,from.y);context.lineTo(to.x,to.y);context.stroke();context.globalAlpha=1;}
function drawBrain(time=0){
  requestAnimationFrame(drawBrain);if(!canvasWidth||!canvasHeight)return;const phase=state.reducedMotion?0:(Math.sin(time/500)+1)/2;
  context.clearRect(0,0,canvasWidth,canvasHeight);
  const glow=context.createRadialGradient(canvasWidth*.5,canvasHeight*.42,15,canvasWidth*.5,canvasHeight*.42,canvasWidth*.45);glow.addColorStop(0,'rgba(118,105,230,.10)');glow.addColorStop(1,'rgba(5,12,20,0)');context.fillStyle=glow;context.fillRect(0,0,canvasWidth,canvasHeight);
  context.strokeStyle='rgba(132,120,218,.12)';context.lineWidth=1;context.beginPath();context.ellipse(canvasWidth*.35,canvasHeight*.4,canvasWidth*.26,canvasHeight*.27,0,0,Math.PI*2);context.ellipse(canvasWidth*.65,canvasHeight*.4,canvasWidth*.26,canvasHeight*.27,0,0,Math.PI*2);context.stroke();
  for(let index=0;index<340;index++)node(kcPoint(index*23),.7,'#756bb0',.22);
  for(let index=0;index<95;index++)node(pnPoint(index*31),.8,'#4a92a8',.25);
  if(!state.trace){node({x:canvasWidth*.5,y:canvasHeight*.54},5+phase*3,'#9c91ff',.55);return;}
  const trace=state.trace;
  if(state.stage===0){const point={x:canvasWidth*.5,y:canvasHeight*.82};node(point,8+phase*5,'#53e6d0',.8);context.strokeStyle=`rgba(83,230,208,${.2+phase*.25})`;context.beginPath();context.arc(point.x,point.y,18+phase*18,0,Math.PI*2);context.stroke();}
  if(state.stage>=1){const max=Math.max(...trace.pn_values.map(Number),1e-9);trace.pn_indices.forEach((index,itemIndex)=>node(pnPoint(index),1.4+2.4*Number(trace.pn_values[itemIndex])/max,'#71dff4',.55+.35*phase));}
  if(state.stage===2){const max=Math.max(state.kcSample[0]?.value||0,1e-9);for(const item of state.kcSample)node(kcPoint(item.index),1+2.2*item.value/max,'#a99fff',.35+.35*item.value/max);}
  if(state.stage>=3){for(const [pn,kc] of state.edgeSample)line(pnPoint(pn),kcPoint(kc),'#60d8ca',.08+.06*phase);for(const index of trace.winner_indices)node(kcPoint(index),2.2+phase*1.3,'#7dffd0',.88);}
  if(state.stage>=4){trace.history.forEach((match,index)=>{const point={x:canvasWidth*(.82+.035*(index%2)),y:canvasHeight*(.22+index*.105)};line(kcPoint(trace.winner_indices[(index*11)%trace.winner_indices.length]),point,'#9c91ff',.22);node(point,3+Number(match.similarity)*3,'#b5adff',.75);});}
  if(state.stage>=5){const point={x:canvasWidth*.5,y:canvasHeight*.16};node(point,7+phase*3,'#f4b66f',.85);context.strokeStyle=`rgba(244,182,111,${.18+phase*.22})`;context.beginPath();context.arc(point.x,point.y,18+phase*11,0,Math.PI*2);context.stroke();}
}
requestAnimationFrame(drawBrain);

for(const button of document.querySelectorAll('.source-tabs [data-source]'))button.addEventListener('click',()=>setSource(button.dataset.source));
$('load-example').addEventListener('click',()=>{const example={alert_name:'Suspicious PowerShell activity',alert_id:'MANUAL-001',severity:'high',category:'execution',mitre_tactic:'Execution',mitre_technique:'T1059.001',hostname:'WS-0142',username:'analyst.user',source_ip:'10.20.1.15',destination_ip:'10.20.8.40',process_name:'powershell.exe',action:'allowed',process_command_line:'powershell.exe -NoProfile -EncodedCommand SQBFAFgA'};for(const input of document.querySelectorAll('[data-field]'))input.value=example[input.dataset.field]||'';$('input-status').textContent='Example loaded';});
$('analyze-alert').addEventListener('click',analyzeAlert);
$('import-row').addEventListener('change',()=>renderImportPreview(Number($('import-row').value)));
$('alert-file').addEventListener('change',async event=>{try{await importFile(event.target.files[0]);}catch(error){showError(error.message);$('input-status').textContent='Import failed';}});
for(const eventName of ['dragenter','dragover'])$('drop-zone').addEventListener(eventName,event=>{event.preventDefault();$('drop-zone').classList.add('dragging');});
for(const eventName of ['dragleave','drop'])$('drop-zone').addEventListener(eventName,event=>{event.preventDefault();$('drop-zone').classList.remove('dragging');});
$('drop-zone').addEventListener('drop',async event=>{try{await importFile(event.dataTransfer.files[0]);}catch(error){showError(error.message);$('input-status').textContent='Import failed';}});
$('analysis-prev').addEventListener('click',()=>{stopReplay();renderStage(state.stage-1);});$('analysis-next').addEventListener('click',()=>{stopReplay();renderStage(state.stage+1);});$('analysis-replay').addEventListener('click',()=>{if(state.playing)return stopReplay();if(state.stage>=STAGES.length-1)renderStage(0);state.playing=true;renderStage(state.stage);queueReplay();});

async function init(){try{state.status=await api('/api/status');$('connection').textContent='Python engine connected';$('input-status').textContent='Ready to analyze';$('analysis-state').textContent='READY';}catch(error){$('connection').textContent='Initialization failed';$('analysis-state').textContent='OFFLINE';showError(error.message);}}
init();
