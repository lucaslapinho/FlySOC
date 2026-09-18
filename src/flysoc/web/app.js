import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const $ = id => document.getElementById(id);
const fmt = value => Number(value).toLocaleString('en-US');
const colors = {optic:'#438dca',central:'#a69ef1',sensory:'#4bc6aa',visual_projection:'#e2af73',ascending:'#7987d6',descending:'#e780a2',sensory_ascending:'#6cd4d4',visual_centrifugal:'#e2be71',motor:'#ed888a',endocrine:'#e2dd96',unclassified:'#7e94ad'};
const JOURNEY_STAGES = [
  {label:'ALERT',title:'Alert received',description:'The selected SOC alert is loaded as inert telemetry. No command, path or payload is executed.'},
  {label:'PN',title:'PN representation',description:'Categorical, text and numeric fields become a sparse engineered feature vector.'},
  {label:'KC',title:'KC expansion',description:'Sparse random connectivity expands PN features into Kenyon Cell-inspired activations.'},
  {label:'TOP-K',title:'Top-K fingerprint',description:'Winner-Take-All keeps the strongest positive activations as a sparse binary fingerprint.'},
  {label:'MEMORY',title:'Associative memory',description:'The fingerprint is compared with historical alerts and its nearest-neighbor novelty is measured.'},
  {label:'HYPOTHESIS',title:'Decision hypothesis',description:'A separate downstream classifier estimates an analyst verdict for review.'}
];
const state = {mode:'real',playing:false,busy:false,epoch:0,count:0,frame:0,catalog:[],filtered:[],trace:null,pulse:null,chart:[],status:null,real:null,fly:null,journeyStage:0,journeyPlaying:false};
let timer, journeyTimer, sceneObjects=[], basePoints, activePoints, graphLines, signalLines;
const stage=$('stage');
let renderer;
try { renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,powerPreference:'high-performance'}); }
catch(error){ $('loading').hidden=true; showError('WebGL is unavailable in this browser. Open the local address in a browser that supports WebGL2.'); throw error; }
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setClearColor(0x000000,0);
stage.prepend(renderer.domElement);
const scene=new THREE.Scene();
const camera=new THREE.PerspectiveCamera(48,1,0.1,1800);
const controls=new OrbitControls(camera,renderer.domElement);
controls.enableDamping=true;controls.dampingFactor=.065;controls.minDistance=30;controls.maxDistance=650;controls.autoRotateSpeed=.4;
function resetCamera(){camera.position.set(0,12,state.mode==='real'?185:200);controls.target.set(0,0,0);controls.update();}
resetCamera();
new ResizeObserver(()=>{const w=stage.clientWidth,h=stage.clientHeight;if(!w||!h)return;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}).observe(stage);
let frames=0,lastFPS=performance.now();
function animate(){requestAnimationFrame(animate);controls.autoRotate=$('auto-rotate').checked;controls.update();renderer.render(scene,camera);frames++;const now=performance.now();if(now-lastFPS>1000){$('fps').textContent=`${Math.round(frames*1000/(now-lastFPS))} FPS`;frames=0;lastFPS=now;}}
animate();

function showError(message){$('error').textContent=message;$('error').hidden=false;setTimeout(()=>$('error').hidden=true,9000);}
async function api(path,body){const response=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await response.json();if(!response.ok)throw new Error(data.error||`HTTP ${response.status}`);return data;}
function addObject(object){scene.add(object);sceneObjects.push(object);return object;}
function clearScene(){for(const item of sceneObjects){scene.remove(item);item.geometry?.dispose();item.material?.dispose();}sceneObjects=[];basePoints=null;activePoints=null;graphLines=null;signalLines=null;}
function points(positions,vertexColors,size,opacity=1){const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setAttribute('color',new THREE.Float32BufferAttribute(vertexColors,3));const material=new THREE.PointsMaterial({size,vertexColors:true,transparent:true,opacity,depthWrite:false,blending:THREE.AdditiveBlending});return new THREE.Points(geometry,material);}
function links(edges,prePositions,postPositions,hex,opacity){const buffer=new Float32Array(edges.length*3);for(let i=0;i<edges.length;i+=2){buffer.set(prePositions.slice(edges[i]*3,edges[i]*3+3),i*3);buffer.set(postPositions.slice(edges[i+1]*3,edges[i+1]*3+3),(i+1)*3);}const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(buffer,3));return new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({color:hex,transparent:true,opacity,depthWrite:false,blending:THREE.AdditiveBlending}));}
function legend(items){$('legend').replaceChildren(...items.map(([color,text])=>{const span=document.createElement('span');const dot=document.createElement('i');dot.style.background=color;span.append(dot,document.createTextNode(text));return span;}));}
function rgb(hex){return new THREE.Color(hex).toArray();}
function drawReal(){clearScene();const data=state.real;const palette=data.group_names.map(name=>rgb(colors[name]||'#7e94ad'));const buffer=new Float32Array(data.positions.length);for(let i=0;i<data.point_nodes.length;i++)buffer.set(palette[data.group_codes[data.point_nodes[i]]],i*3);basePoints=addObject(points(data.positions,buffer,.30,.7));basePoints.material.blending=THREE.NormalBlending;basePoints.material.opacity=.55;basePoints.userData.kind='real';graphLines=addObject(links(data.edges,data.representative_positions,data.representative_positions,'#55b8c7',.003));legend([['#438dca','Optic'],['#a69ef1','Central'],['#4bc6aa','Sensory'],['#ffc76f','Computed signal']]);if(state.pulse)drawPulse(state.pulse);applyVisibility();}
function drawFly(){clearScene();const data=state.fly;const positions=new Float32Array([...data.pn_positions,...data.kc_positions]);const buffer=new Float32Array(positions.length);const pnRGB=rgb('#51b3c9'),kcRGB=rgb('#847aca');for(let i=0;i<positions.length/3;i++)buffer.set(i<state.status.pn_dim?pnRGB:kcRGB,i*3);basePoints=addObject(points(positions,buffer,.52,.35));basePoints.userData.kind='fly';graphLines=addObject(links(data.edges,data.pn_positions,data.kc_positions,'#3d6e86',.028));legend([['#59bcda','PN features'],['#9485d6','KC activations'],['#70efc5','Top-K winners']]);if(state.trace)drawTraceStage(state.trace,state.journeyStage);applyVisibility();}
function removeDynamic(){for(const object of [activePoints,signalLines]){if(object){scene.remove(object);object.geometry.dispose();object.material.dispose();sceneObjects=sceneObjects.filter(item=>item!==object);}}activePoints=null;signalLines=null;}
function drawPulse(frame){removeDynamic();const positions=[],vertexColors=[];const max=Math.max(frame.max_activity,1e-12);for(let j=0;j<frame.node_indices.length;j++){const index=frame.node_indices[j],strength=Math.sqrt(frame.node_values[j]/max);positions.push(...state.real.representative_positions.slice(index*3,index*3+3));const color=new THREE.Color('#ffd791').multiplyScalar(.35+.65*strength);vertexColors.push(...color.toArray());}activePoints=addObject(points(positions,vertexColors,1.45,.95));applyVisibility();}
function drawTraceStage(trace,stageIndex){
  removeDynamic();
  if(stageIndex<1)return applyVisibility();
  const positions=[],vertexColors=[];
  const pnMax=Math.max(...trace.pn_values,1e-12);
  for(let j=0;j<trace.pn_indices.length;j++){
    const index=trace.pn_indices[j];
    positions.push(...state.fly.pn_positions.slice(index*3,index*3+3));
    vertexColors.push(...new THREE.Color('#7de5fb').multiplyScalar(.4+.6*trace.pn_values[j]/pnMax).toArray());
  }
  if(stageIndex===2){
    const active=trace.kc_activations.map((value,index)=>({value,index})).filter(item=>item.value>0).sort((a,b)=>b.value-a.value).slice(0,700);
    const kcMax=Math.max(active[0]?.value||0,1e-12);
    for(const item of active){positions.push(...state.fly.kc_positions.slice(item.index*3,item.index*3+3));vertexColors.push(...new THREE.Color('#a89fff').multiplyScalar(.28+.72*item.value/kcMax).toArray());}
  }
  if(stageIndex>=3){
    for(const index of trace.winner_indices){positions.push(...state.fly.kc_positions.slice(index*3,index*3+3));vertexColors.push(...rgb('#7dffd0'));}
    signalLines=addObject(links(trace.contributing_edges,state.fly.pn_positions,state.fly.kc_positions,'#59d6c2',.22));
  }
  activePoints=addObject(points(positions,vertexColors,stageIndex>=3?1.8:1.25,1));
  applyVisibility();
}
function applyVisibility(){if(basePoints){basePoints.visible=!$('only-active').checked;basePoints.material.size=Number($('point-size').value)*(state.mode==='real'?.20:.35);}if(graphLines)graphLines.visible=$('show-links').checked&&!$('only-active').checked;if(signalLines)signalLines.visible=$('show-links').checked;}

function setStats(labels,values,details){for(let i=0;i<4;i++){$(`stat-label-${i+1}`).textContent=labels[i];$(`stat-${i+1}`).textContent=values[i];$(`stat-detail-${i+1}`).textContent=details[i];}}
function setTraceStrip(labels,selected=0,interactive=false){$('trace-strip').replaceChildren();labels.forEach((label,i)=>{if(i){$('trace-strip').append(document.createElement('i'));}const node=document.createElement(interactive?'button':'span');if(interactive){node.type='button';node.addEventListener('click',()=>{stopJourney();renderJourneyStage(i);});}node.className=`trace-node${i<selected?' complete':''}${i===selected?' selected':''}`;node.textContent=label;$('trace-strip').append(node);});}
function setMode(mode){
  stop();state.epoch++;state.mode=mode;document.body.dataset.mode=mode;state.chart=[];
  const real=mode==='real';
  $('mode-real').setAttribute('aria-selected',String(real));$('mode-fly').setAttribute('aria-selected',String(!real));
  $('real-controls').hidden=!real;$('fly-controls').hidden=real;$('manual-section').hidden=real;$('scene-labels').hidden=real;$('selected-node').hidden=true;
  for(const id of ['journey-controls','journey-stage-card','alert-context','decision-card','fingerprint-card'])$(id).hidden=real;
  $('mode-note').textContent=real?'Real anatomical positions · exploratory propagation':'Model activations computed per alert · schematic layout';
  $('viewer-title').textContent=real?'Drosophila connectome':'Mushroom Body-inspired · FlySOC';
  $('scene-tag').textContent=real?'ANATOMY · ANNOTATED POINTS':'COMPUTATIONAL NETWORK · SCHEMATIC POSITIONS';
  $('geometry-caption').textContent=real?'Lines show connections, not axon trajectories.':'Stages replay one completed Python trace; positions are schematic.';
  $('science-note').textContent=real?'Connections and positions come from FlyWire. The pulse is mathematical diffusion without inhibitory signals, voltages or spikes. It is not a validated physiological simulation.':'PN and KC are engineering components. The staged view replays a completed calculation. The downstream classifier produces a review hypothesis, not an automatic verdict.';
  $('chart-title').textContent=real?'Activity across steps':'KC activation distribution';$('list-heading').textContent=real?'NEURONS WITH THE STRONGEST SIGNAL':'NEAREST HISTORICAL ALERTS';
  $('metric-a-label').textContent=real?'STEP':'NOVELTY';$('metric-b-label').textContent=real?'SIGNAL MASS':'TOP-1 SIMILARITY';
  $('result-badge').className='badge';$('result-badge').textContent='Ready to explore';$('result-title').textContent=real?'A map of real connections.':'Observe the next alert.';$('result-description').textContent=real?'Apply a pulse to follow the computed propagation through the graph.':'Run one step to replay the complete alert decision journey.';
  $('results-list').replaceChildren();$('metric-a').textContent='0';$('metric-b').textContent='0';
  if(real){
    const c=state.status.connectome;setStats(['NEURONS','CONNECTED NEURON PAIRS','ACTIVE SIGNAL','COMPUTE / STEP'],[fmt(c.neurons),fmt(c.neuron_pairs),'0','—'],['FlyWire FAFB v783',`${fmt(c.synapses)} synapses in the dataset`,'Awaiting stimulation','Computed in Python']);
    $('data-caption').textContent=`${fmt(c.coordinate_points)} annotated positions · ${fmt(c.rendered_edges)} connections rendered / ${fmt(c.neuron_pairs)} computed`;drawReal();if(state.pulse)updatePulsePanel(state.pulse);
  }else{
    setStats(['PN FEATURES','KENYON CELLS','ACTIVE TOP-K','COMPUTE / ALERT'],[fmt(state.status.pn_dim),fmt(state.status.kc_dim),'0','—'],['Hash + TF-IDF + numeric features',`Fan-in ${state.status.fan_in} · random expansion`,`Limit of ${state.status.top_k} winners`,'PN + KC + memory + classifier']);
    $('data-caption').textContent=`${fmt(state.status.training_alerts)} alerts in memory · ${fmt(state.status.test_alerts)} test alerts`;drawFly();if(state.trace)updateTracePanel(state.trace);else setTraceStrip(JOURNEY_STAGES.map(item=>item.label),0,false);
  }
  if(real)setTraceStrip(['PULSE','REAL GRAPH','DIFFUSION','READOUT']);
  resetCamera();updateFrameLabel();chart();
}

function chart(values=state.chart){const canvas=$('activity-chart'),ctx=canvas.getContext('2d');const w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);ctx.strokeStyle='#223549';ctx.lineWidth=1;for(let y=20;y<h;y+=35){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}if(!values.length)return;const max=Math.max(...values,1e-12),accent=getComputedStyle(document.body).getPropertyValue('--accent').trim()||'#53e6d0';ctx.strokeStyle=accent;ctx.fillStyle=`${accent}30`;ctx.lineWidth=2;ctx.beginPath();values.forEach((v,i)=>{const x=i*w/Math.max(1,values.length-1),y=h-12-v/max*(h-26);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();ctx.lineTo(w,h);ctx.lineTo(0,h);ctx.closePath();ctx.fill();}
function rows(items,kind){$('results-list').replaceChildren(...items.map((item,i)=>{const row=document.createElement('div');row.className='result-row';const rank=document.createElement('span');rank.className='rank';rank.textContent=String(i+1).padStart(2,'0');const info=document.createElement('div');info.className='match-info';const title=document.createElement('strong');title.textContent=kind==='pulse'?item.class:item.alert_name;const sub=document.createElement('small');sub.textContent=kind==='pulse'?item.root_id:`${item.alert_id} · ${item.verdict||'UNLABELED'}`;info.append(title,sub);const score=document.createElement('span');score.className='score';score.textContent=kind==='pulse'?Number(item.activity).toFixed(3):`${(item.similarity*100).toFixed(1)}%`;row.append(rank,info,score);return row;}));}
function emptyResults(message){const item=document.createElement('p');item.className='empty';item.textContent=message;$('results-list').replaceChildren(item);}
function readable(value){return String(value||'').replaceAll('_',' ').replace(/\b\w/g,character=>character.toUpperCase());}
function renderAlertContext(trace){
  const fields=[['ALERT ID','alert_id'],['SEVERITY','severity'],['CATEGORY','category'],['PROCESS','process_name'],['HOST','hostname'],['MITRE','mitre_technique']];
  const items=fields.filter(([,key])=>trace.alert[key]!==undefined&&trace.alert[key]!==null&&String(trace.alert[key]).trim()).map(([label,key])=>{const item=document.createElement('div');item.className='context-item';const name=document.createElement('span');name.textContent=label;const value=document.createElement('strong');value.textContent=String(trace.alert[key]);value.title=String(trace.alert[key]);item.append(name,value);return item;});
  $('alert-context-grid').replaceChildren(...items);$('alert-context').hidden=false;
}
function renderFingerprint(trace){
  $('fingerprint-summary').textContent=`${fmt(trace.active_kcs)} / ${fmt(state.status.kc_dim)} ACTIVE`;
  $('fingerprint-grid').replaceChildren(...trace.winner_indices.map(index=>{const dot=document.createElement('span');dot.className='fingerprint-dot';dot.title=`Active KC ${index}`;dot.setAttribute('aria-label',`Active KC ${index}`);return dot;}));
}
function renderDecision(trace){
  const prediction=trace.prediction;
  if(!prediction){$('decision-probabilities').replaceChildren();$('decision-model').textContent='No downstream classifier was saved with this experiment.';return;}
  const entries=Object.entries(prediction.probabilities||{}).sort((a,b)=>b[1]-a[1]);
  const confidence=prediction.confidence??entries[0]?.[1]??0;
  $('decision-hypothesis').textContent=readable(prediction.hypothesis||prediction.verdict||'Unknown');
  $('decision-confidence').textContent=`${(confidence*100).toFixed(1)}%`;
  $('decision-model').textContent=`Model: ${prediction.model} · separate from the FlySOC representation`;
  $('decision-probabilities').replaceChildren(...entries.map(([label,value])=>{const row=document.createElement('div');row.className='probability-row';const name=document.createElement('span');name.textContent=readable(label);const track=document.createElement('div');track.className='probability-track';const fill=document.createElement('i');fill.style.width=`${Math.max(0,Math.min(100,value*100))}%`;track.append(fill);const score=document.createElement('strong');score.textContent=`${(value*100).toFixed(1)}%`;row.append(name,track,score);return row;}));
}
function renderJourneyStage(index){
  if(state.mode!=='fly'||!state.trace)return;
  state.journeyStage=Math.max(0,Math.min(JOURNEY_STAGES.length-1,index));
  const stageInfo=JOURNEY_STAGES[state.journeyStage],trace=state.trace;
  $('journey-stage-card').hidden=false;$('journey-controls').hidden=false;
  $('journey-stage-number').textContent=`STEP ${state.journeyStage+1} / ${JOURNEY_STAGES.length}`;$('journey-stage-title').textContent=stageInfo.title;$('journey-stage-description').textContent=stageInfo.description;
  $('journey-position').textContent=`${String(state.journeyStage+1).padStart(2,'0')} / ${String(JOURNEY_STAGES.length).padStart(2,'0')}`;
  $('journey-prev').disabled=state.journeyStage===0;$('journey-next').disabled=state.journeyStage===JOURNEY_STAGES.length-1;$('journey-play').textContent=state.journeyPlaying?'Ⅱ Pause':'▶ Replay';
  setTraceStrip(JOURNEY_STAGES.map(item=>item.label),state.journeyStage,true);drawTraceStage(trace,state.journeyStage);
  $('alert-context').hidden=false;$('fingerprint-card').hidden=state.journeyStage<3;$('decision-card').hidden=state.journeyStage<5||!trace.prediction;
  $('result-title').textContent=trace.alert.alert_name||trace.alert.alert_id||'Alert';$('result-description').textContent=stageInfo.description;
  if(state.journeyStage<4){$('result-badge').textContent=stageInfo.title;$('result-badge').className='badge';emptyResults('Historical matches appear when the journey reaches associative memory.');}
  else{$('result-badge').textContent=trace.novelty.is_novel?'Unfamiliar pattern':'Recurrent pattern';$('result-badge').className=`badge${trace.novelty.is_novel?' novel':''}`;rows(trace.history,'trace');}
  $('stat-3').textContent=state.journeyStage>=3?fmt(trace.active_kcs):(state.journeyStage===2?fmt(trace.nonzero_activations):'—');
  $('stat-detail-3').textContent=state.journeyStage>=3?`${fmt(trace.active_kcs)} Top-K winners`:state.journeyStage===2?`${fmt(trace.nonzero_activations)} KCs with positive activation`:'Awaiting KC activation';
}
function journeyDelay(){return Math.max(300,1000/Number($('speed').value));}
function stopJourney(){clearTimeout(journeyTimer);state.journeyPlaying=false;if($('journey-play'))$('journey-play').textContent='▶ Replay';}
function queueJourney(){clearTimeout(journeyTimer);journeyTimer=setTimeout(()=>{if(!state.journeyPlaying)return;if(state.journeyStage>=JOURNEY_STAGES.length-1){stopJourney();return;}renderJourneyStage(state.journeyStage+1);queueJourney();},journeyDelay());}
function startJourney(){stopJourney();state.journeyPlaying=true;renderJourneyStage(0);queueJourney();}
function updatePulsePanel(frame){$('stat-3').textContent=fmt(frame.active_neurons);$('stat-detail-3').textContent='Amplitude > 10⁻⁸ across the full graph';$('stat-4').textContent=`${frame.compute_ms.toFixed(1)} ms`;$('result-badge').textContent='Computed diffusion';$('result-badge').className='badge';$('result-title').textContent=`Pulse in ${frame.source_class}`;$('result-description').textContent=`${fmt(frame.display_limit)} highest amplitudes can be highlighted. Computation includes all neurons and connections in the dataset.`;$('metric-a').textContent=String(frame.iteration);$('metric-b').textContent=frame.activity_mass.toFixed(2);$('chart-scale').textContent='total mass · relative scale';rows(frame.leaders,'pulse');state.chart.push(frame.activity_mass);state.chart=state.chart.slice(-80);chart();setTraceStrip(['PULSE','REAL GRAPH','DIFFUSION','READOUT'],frame.iteration?2:0);}
function updateTracePanel(trace){
  $('stat-4').textContent=`${trace.timing_ms.total.toFixed(1)} ms`;$('metric-a').textContent=trace.novelty.novelty_score.toFixed(3);$('metric-b').textContent=`${((trace.history[0]?.similarity||0)*100).toFixed(1)}%`;$('chart-scale').textContent=`novelty threshold ${trace.novelty_threshold.toFixed(3)}`;
  renderAlertContext(trace);renderFingerprint(trace);renderDecision(trace);
  const bins=new Array(60).fill(0),max=Math.max(...trace.kc_activations,1e-12);for(const value of trace.kc_activations)if(value>0)bins[Math.min(59,Math.floor(value/max*59))]++;chart(bins);
  startJourney();
}
function updateFrameLabel(){const index=state.mode==='real'?(state.pulse?.iteration||0):state.frame;$('frame-label').textContent=`${state.playing?'RUNNING':'PAUSED'} · ${state.mode==='real'?'STEP':'ALERT'} ${index}`;$('play').textContent=state.playing?'Ⅱ Pause':'▶ Start';}
function stop(){state.playing=false;clearTimeout(timer);stopJourney();updateFrameLabel();}
function schedule(){clearTimeout(timer);if(state.playing){const delay=state.mode==='fly'?JOURNEY_STAGES.length*journeyDelay()+120:1000/Number($('speed').value);timer=setTimeout(async()=>{await step();schedule();},delay);}}
async function step(resetClass=null,manual=null){if(state.busy)return;state.busy=true;$('step').disabled=true;const epoch=state.epoch;try{if(state.mode==='real'){const body=resetClass?{reset_class:resetClass,advance:false}:state.pulse?{advance:true}:{reset_class:$('stimulus').value,advance:false};const frame=await api('/api/connectome/pulse',body);if(epoch!==state.epoch)return;state.pulse=frame;if(resetClass)state.chart=[];drawPulse(frame);updatePulsePanel(frame);}else{const index=Number($('alert-select').value);if(!manual&&!Number.isInteger(index))throw new Error('Select an alert.');const trace=await api('/api/fly/trace',manual?{alert:manual}:{index});if(epoch!==state.epoch)return;state.trace=trace;state.frame++;updateTracePanel(trace);if(state.playing&&!manual){const position=state.filtered.findIndex(item=>item.index===index);$('alert-select').value=String(state.filtered[(position+1)%state.filtered.length].index);}}state.count++;$('processed-count').textContent=String(state.count);updateFrameLabel();}catch(error){showError(error.message);stop();}finally{state.busy=false;$('step').disabled=false;}}
function filterAlerts(){const filter=$('alert-filter').value;state.filtered=state.catalog.filter(item=>filter==='all'||(filter==='unseen'?item.unseen:item.verdict===filter));$('alert-select').replaceChildren(...state.filtered.map(item=>{const option=document.createElement('option');option.value=item.index;option.textContent=`${item.alert_id} · ${item.alert_name}`;return option;}));}
$('mode-real').addEventListener('click',()=>setMode('real'));$('mode-fly').addEventListener('click',()=>setMode('fly'));
$('play').addEventListener('click',async()=>{if(state.playing)return stop();state.playing=true;updateFrameLabel();await step();schedule();});
$('step').addEventListener('click',()=>step());$('pulse').addEventListener('click',()=>step($('stimulus').value));
$('journey-prev').addEventListener('click',()=>{stopJourney();renderJourneyStage(state.journeyStage-1);});
$('journey-next').addEventListener('click',()=>{stopJourney();renderJourneyStage(state.journeyStage+1);});
$('journey-play').addEventListener('click',()=>{if(state.journeyPlaying)return stopJourney();if(state.journeyStage>=JOURNEY_STAGES.length-1)renderJourneyStage(0);state.journeyPlaying=true;renderJourneyStage(state.journeyStage);queueJourney();});
$('alert-filter').addEventListener('change',()=>{stop();filterAlerts();});$('alert-select').addEventListener('change',()=>{stop();step();});
$('speed').addEventListener('input',()=>{$('speed-label').value=`${$('speed').value}×`;schedule();});
for(const id of ['show-links','only-active','point-size'])$(id).addEventListener('input',()=>{applyVisibility();$('point-size-label').value=Number($('point-size').value).toLocaleString('en-US');});
$('reset-view').addEventListener('click',resetCamera);
$('expand-view').addEventListener('click',()=>{const expanded=document.querySelector('.viewer').classList.toggle('expanded');$('expand-view').setAttribute('aria-label',expanded?'Collapse viewer':'Expand viewer');});
document.addEventListener('keydown',event=>{if(event.key==='Escape'){document.querySelector('.viewer').classList.remove('expanded');$('expand-view').setAttribute('aria-label','Expand viewer');}});
$('manual-process').addEventListener('click',()=>{try{stop();const alert=JSON.parse($('manual-alert').value);if(!alert||typeof alert!=='object'||Array.isArray(alert))throw new Error('Provide a JSON alert object.');step(null,alert);}catch(error){showError(`Invalid JSON: ${error.message}`);}});
const raycaster=new THREE.Raycaster();raycaster.params.Points.threshold=.6;let pointerDown;
renderer.domElement.addEventListener('pointerdown',event=>pointerDown=[event.clientX,event.clientY]);
renderer.domElement.addEventListener('pointerup',async event=>{if(!pointerDown||Math.hypot(event.clientX-pointerDown[0],event.clientY-pointerDown[1])>5)return;const rect=renderer.domElement.getBoundingClientRect();raycaster.setFromCamera(new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1),camera);const hits=basePoints?.visible?raycaster.intersectObject(basePoints):[];if(!hits.length)return;const i=hits[0].index;$('selected-node').hidden=false;if(state.mode==='real'){const node=state.real.point_nodes[i],group=state.real.group_names[state.real.group_codes[node]];try{const info=await api(`/api/connectome/neuron/${node}`);$('selected-node').textContent=`Root ID ${info.root_id} · ${info.class} · ${group} · predicted neurotransmitter ${info.nt_type}`;}catch(error){showError(error.message);}}else{const pn=i<state.status.pn_dim,index=pn?i:i-state.status.pn_dim;const value=pn?(state.trace?.pn_values[state.trace.pn_indices.indexOf(index)]||0):(state.trace?.kc_activations[index]||0);$('selected-node').textContent=`${pn?'PN':'KC'} ${index} · activation ${value.toFixed(5)}${!pn&&state.trace?.winner_indices.includes(index)?' · Top-K winner':''}`;}});

async function init(){try{state.status=await api('/api/status');if(!state.status.connectome)throw new Error('Run download_connectome.py and prepare_connectome.py to load the real data.');const [real,fly,catalog]=await Promise.all([api('/api/connectome/scene'),api('/api/fly/layout'),api('/api/alerts')]);state.real=real;state.fly=fly;state.catalog=catalog;const names={ALPN:'ALPN · olfactory projection',Kenyon_Cell:'Kenyon Cells · Mushroom Body',MBON:'MBON · Mushroom Body output',DAN:'DAN · dopaminergic neurons',visual:'Visual sensory',olfactory:'Olfactory sensory'};$('stimulus').replaceChildren(...state.status.stimulus_classes.map(name=>{const option=document.createElement('option');option.value=name;option.textContent=names[name]||name;return option;}));filterAlerts();setMode('real');$('loading').hidden=true;$('connection').textContent='Python engine connected';}catch(error){$('loading').hidden=true;$('connection').textContent='Initialization failed';showError(error.message);console.error(error);}}
await init();
