import {forward} from './inference.js';
const $=id=>document.getElementById(id);
const percent=x=>(100*x).toFixed(1)+'%';
let results,model,run,selected,output,threshold,revealed=false,filter='all',inspectLayer,inspectChannel=0,inputChannel=0,pixel={y:0,x:0};
const size=28;
function paint(canvas,layer,channel=0,gray=false){
  canvas.width=layer.w;canvas.height=layer.h;
  const ctx=canvas.getContext('2d'),im=ctx.createImageData(layer.w,layer.h);
  const arr=layer.data.subarray(channel*layer.h*layer.w,(channel+1)*layer.h*layer.w);
  let scale=1e-6;for(const v of arr)scale=Math.max(scale,Math.abs(v));
  for(let i=0;i<arr.length;i++){
    let rgb;
    if(gray){const v=Math.round((arr[i]+1)*127.5);rgb=[v,v,v];}
    else{const v=arr[i]/scale;const color=v>=0?[201,89,71]:[58,116,159];rgb=color.map(c=>Math.round(248+(c-248)*Math.abs(v)));}
    im.data.set([...rgb,255],i*4);
  }
  ctx.putImageData(im,0,0);
}
async function loadImage(id){
  const img=new Image();img.src=`assets/samples/${id}-${size}.png`;await img.decode();
  const c=document.createElement('canvas');c.width=c.height=size;
  const ctx=c.getContext('2d');ctx.drawImage(img,0,0);const rgba=ctx.getImageData(0,0,size,size).data;
  return Uint8Array.from({length:size*size},(_,i)=>rgba[i*4]);
}
function drawGallery(){
  const cases=results.cases.filter(c=>filter==='all'||(filter==='fp'?c.label===0&&c.scores[size]>=threshold:c.label===1&&c.scores[size]<threshold));
  $('case-count').textContent=`${cases.length} cases`;
  $('gallery').replaceChildren();
  if(!cases.length){$('gallery').textContent='No examples in this gallery at the current threshold.';return;}
  for(const c of cases){
    const b=document.createElement('button');b.className='case-thumb'+(selected?.id===c.id?' selected':'');
    b.setAttribute('aria-label',`Select test image ${c.id}`);b.setAttribute('aria-pressed',selected?.id===c.id?'true':'false');
    const img=document.createElement('img');img.src=`assets/samples/${c.id}-${size}.png`;img.alt=`Test image ${c.id}`;
    const text=document.createElement('span');text.textContent=`#${c.id}`;b.append(img,text);b.onclick=()=>selectCase(c);$('gallery').append(b);
  }
}
async function selectCase(c){
  selected=c;revealed=false;$('truth').textContent='';$('reveal').textContent='Reveal dataset label';
  document.querySelectorAll('.guess-row button').forEach(b=>b.classList.remove('selected'));
  $('prediction').textContent='Computing…';
  try{
    const pixels=await loadImage(c.id);const tick=performance.now();const nextOutput=forward(pixels,model);const ms=performance.now()-tick;
    if(selected.id!==c.id)return;
    output=nextOutput;
    $('runtime').textContent=`${ms.toFixed(1)} ms · browser inference`;
    paint($('case-image'),output.layers[0],0,true);paint($('hero-image'),output.layers[0],0,true);
    $('case-id').textContent=`Test image #${c.id} · 28 × 28`;
    updatePrediction();drawGallery();drawNetwork();
  }catch(e){$('prediction').textContent='Unable to load image';$('truth').textContent=e.message;}
}
function updatePrediction(){
  if(!output)return;
  const score=output.score;
  $('prediction').textContent=score>=threshold?'Pneumonia-leaning':'Normal-leaning';
  $('score').textContent=percent(score);$('hero-score').textContent=percent(score);
  $('score-fill').style.width=percent(score);$('threshold-tick').style.left=percent(threshold);
  if(revealed)showTruth();
}
function showTruth(){
  revealed=true;const correct=(output.score>=threshold)==selected.label;
  $('truth').innerHTML=`<span class="result-pill ${correct?'':'error'}">Dataset label: ${selected.label?'Pneumonia':'Normal'} · ${correct?'Matched':'Mismatched'}</span>`;
  $('reveal').textContent='Label revealed';
}
function drawNetwork(){
  $('network').replaceChildren();
  for(const layer of output.layers){
    const col=document.createElement('div');col.className='layer';
    const title=document.createElement('div');title.className='layer-title';title.textContent=layer.label;
    const shape=document.createElement('div');shape.className='layer-shape';shape.textContent=`${layer.h} × ${layer.w} · ${layer.c} ch`;
    col.append(title,shape);
    const mapBox=document.createElement('div');col.append(mapBox);
    let count=Math.min(4,layer.c);
    function render(){mapBox.replaceChildren();for(let ch=0;ch<count;ch++){
      const b=document.createElement('button');b.className='map-button';b.setAttribute('aria-label',`Inspect ${layer.label} channel ${ch}`);
      const c=document.createElement('canvas');paint(c,layer,ch,layer.kind==='input');
      const label=document.createElement('span');label.textContent=layer.c>1?`Channel ${ch}`:'Grayscale';b.append(c,label);b.onclick=()=>openInspector(layer,ch);mapBox.append(b);
    }}render();
    if(layer.c>4){const b=document.createElement('button');b.className='layer-more';b.textContent=`All ${layer.c} channels`;b.onclick=()=>{count=count===4?layer.c:4;b.textContent=count===4?`All ${layer.c} channels`:'Show fewer';render();};col.append(b);}
    $('network').append(col);
  }
  const col=document.createElement('div');col.className='layer';col.innerHTML='<div class="layer-title">Average + head</div><div class="layer-shape">24 values → 1 score</div>';
  const b=document.createElement('button');b.className='output-node';b.innerHTML=`<strong>${percent(output.score)}</strong><span>Inspect contributions</span>`;b.onclick=()=>openInspector({kind:'head',label:'Average + head',c:1,h:1,w:1},0);col.append(b);$('network').append(col);
}
function options(select,n,value){select.replaceChildren();for(let i=0;i<n;i++){const o=document.createElement('option');o.value=i;o.textContent=i;select.append(o);}select.value=value;}
function openInspector(layer,ch){
  inspectLayer=layer;inspectChannel=ch;inputChannel=0;pixel={y:Math.floor(layer.h/2),x:Math.floor(layer.w/2)};
  options($('channel-select'),layer.c,ch);
  const parent=output.layers.find(l=>l.id===layer.parent);
  options($('input-channel-select'),parent?.c||1,0);
  $('input-channel-label').hidden=layer.kind!=='conv';
  $('channel-select').parentElement.hidden=layer.kind==='head';
  refreshInspector();$('inspector').showModal();
}
function grid(values,columns,maximum=false){
  const el=document.createElement('div');el.className='math-grid';el.style.gridTemplateColumns=`repeat(${columns},1fr)`;
  const max=Math.max(...values);
  values.forEach(v=>{const cell=document.createElement('div');cell.textContent=v.toFixed(3);if(maximum&&v===max)cell.className='maximum';el.append(cell);});return el;
}
function value(layer,ch,y,x){return y<0||x<0||y>=layer.h||x>=layer.w?0:layer.data[ch*layer.h*layer.w+y*layer.w+x];}
function refreshInspector(){
  const l=inspectLayer,p=output.layers.find(a=>a.id===l.parent),ch=inspectChannel,{y,x}=pixel;
  $('inspect-title').textContent=l.label+(l.c>1?` · channel ${ch}`:'');
  $('math-panel').replaceChildren();$('math-result').textContent='';
  $('inspect-map').parentElement.hidden=l.kind==='head';
  const descriptions={input:'The downloaded grayscale values are scaled from 0–255 to −1–1. No further resizing is applied.',conv:'A learned 3 × 3 filter combines a local patch from every input channel, then adds a bias. Select an input channel to inspect its contribution. Zero padding keeps the map size unchanged.',relu:'ReLU keeps positive values and replaces negative values with zero. It has no learned weights.',pool:'Max pooling takes the largest value in each 2 × 2 block. It halves the map dimensions without learning any weights.',head:'Each of the 24 final maps is averaged. A learned weight multiplies each average, a bias is added, and a sigmoid converts the resulting logit to a score.'};
  $('inspect-description').textContent=descriptions[l.kind];
  if(l.kind!=='head'){
    paint($('inspect-map'),l,ch,l.kind==='input');
    const ctx=$('inspect-map').getContext('2d');ctx.strokeStyle='#f4bc37';ctx.lineWidth=.18;ctx.strokeRect(x+.1,y+.1,.8,.8);
    $('pixel-location').textContent=`Selected pixel: row ${y}, column ${x}`;
  }
  const addTitle=t=>{const h=document.createElement('h4');h.textContent=t;$('math-panel').append(h);};
  if(l.kind==='conv'){
    const w=model.weights[l.weight+'.weight'],b=model.weights[l.weight+'.bias'].values[ch];
    const patch=[],kernel=[];
    for(let dy=0;dy<3;dy++)for(let dx=0;dx<3;dx++){
      patch.push(value(p,inputChannel,y+dy-1,x+dx-1));kernel.push(w.values[((ch*p.c+inputChannel)*3+dy)*3+dx]);
    }
    const partial=patch.reduce((sum,v,i)=>sum+v*kernel[i],0);
    addTitle(`Input patch · channel ${inputChannel}`);$('math-panel').append(grid(patch,3));
    addTitle('Learned kernel');$('math-panel').append(grid(kernel,3));
    $('math-result').textContent=`This channel’s dot product: ${partial.toFixed(5)}. Sum across ${p.c} input channel${p.c===1?'':'s'} + bias (${b.toFixed(5)}) = ${value(l,ch,y,x).toFixed(5)}. Values shown above are rounded.`;
  }else if(l.kind==='pool'){
    const patch=[value(p,ch,2*y,2*x),value(p,ch,2*y,2*x+1),value(p,ch,2*y+1,2*x),value(p,ch,2*y+1,2*x+1)];
    addTitle('Input block · maximum highlighted');$('math-panel').append(grid(patch,2,true));$('math-result').textContent=`max(${patch.map(v=>v.toFixed(4)).join(', ')}) = ${value(l,ch,y,x).toFixed(4)}`;
  }else if(l.kind==='relu'){
    addTitle('Before → after');$('math-panel').append(grid([value(p,ch,y,x),value(l,ch,y,x)],2));$('math-result').textContent=`max(0, ${value(p,ch,y,x).toFixed(5)}) = ${value(l,ch,y,x).toFixed(5)}`;
  }else if(l.kind==='input'){
    const v=value(l,0,y,x),raw=Math.round((v+1)*127.5);addTitle('Grayscale → normalized');$('math-panel').append(grid([raw,v],2));$('math-result').textContent=`${raw} ÷ 127.5 − 1 = ${v.toFixed(5)}`;
  }else{
    const list=document.createElement('div');list.className='contribution-list';
    output.contributions.forEach((v,i)=>{const row=document.createElement('div');row.innerHTML=`<span>Ch ${i}: ${output.means[i].toFixed(3)} × ${model.weights['head.weight'].values[i].toFixed(3)}</span><strong>${v.toFixed(4)}</strong>`;list.append(row);});$('math-panel').append(list);
    $('math-result').textContent=`Sum of contributions + bias (${model.weights['head.bias'].values[0].toFixed(5)}) = logit ${output.logit.toFixed(5)}. Sigmoid(logit) = ${output.score.toFixed(5)}. Contributions are mathematical weights, not clinical explanations.`;
  }
}
function updateMetrics(){
  const y=results.test_labels,p=results.test_scores[size];let tp=0,tn=0,fp=0,fn=0;
  y.forEach((v,i)=>{if(v===1){if(p[i]>=threshold)tp++;else fn++;}else{if(p[i]>=threshold)fp++;else tn++;}});
  $('threshold-value').textContent=threshold.toFixed(3);$('sensitivity').textContent=percent(tp/(tp+fn));$('specificity').textContent=percent(tn/(tn+fp));$('precision').textContent=tp+fp?percent(tp/(tp+fp)):'—';$('auc').textContent=run.test.auc.toFixed(3);
  $('auc-ci').textContent=`95% image-bootstrap interval: ${run.test_auc_ci.map(v=>v.toFixed(3)).join('–')}`;
  $('matrix').innerHTML=`<div><span>Normal → normal</span><strong>${tn}</strong></div><div class="error"><span>Normal → pneumonia</span><strong>${fp}</strong></div><div class="error"><span>Pneumonia → normal</span><strong>${fn}</strong></div><div><span>Pneumonia → pneumonia</span><strong>${tp}</strong></div>`;
  updatePrediction();drawGallery();
}
function learningCurve(){
  const history=run.history,points=history.map((r,i)=>`${25+i/(history.length-1)*375},${125-(r.val_auc-.9)/.1*100}`);
  const best=history.findIndex(r=>r.epoch===run.best_epoch),bx=25+best/(history.length-1)*375,by=125-(history[best].val_auc-.9)/.1*100;
  $('learning-curve').innerHTML=`<line x1="25" y1="125" x2="400" y2="125" stroke="#d5dfe3"/><line x1="25" y1="25" x2="400" y2="25" stroke="#e7edef"/><text x="0" y="130" font-size="9" fill="#70828c">.90</text><text x="0" y="29" font-size="9" fill="#70828c">1.0</text><polyline points="${points.join(' ')}" fill="none" stroke="#365ee7" stroke-width="2.5"/><circle cx="${bx}" cy="${by}" r="4" fill="#16695d"/><text x="25" y="146" font-size="9" fill="#70828c">Epoch 1</text><text x="347" y="146" font-size="9" fill="#70828c">Epoch ${history.length}</text>`;
  $('training-note').textContent=`Selected epoch ${run.best_epoch} by validation AUC. Training: ${run.training_seconds.toFixed(1)} seconds on ${results.device.toUpperCase()}.`;
}
async function init(){
  try{
    [results,model]=await Promise.all(['assets/results.json','assets/model-28.json'].map(async url=>{const r=await fetch(url);if(!r.ok)throw Error(`Could not load ${url}`);return r.json();}));
    run=results.runs.find(r=>r.seed===42&&r.size===28);threshold=run.threshold;$('threshold').value=threshold;
    $('class-balance').textContent=Object.entries(results.split_counts).map(([k,v])=>`${k}: ${v.normal} normal / ${v.pneumonia} pneumonia`).join(' · ');
    $('training-details').textContent=`${run.parameters.toLocaleString()} parameters; ${run.history.length} epochs run; selected epoch ${run.best_epoch}. PyTorch ${results.torch_version}.`;
    $('parity-note').textContent=`ONNX vs PyTorch maximum logit difference: ${run.onnx_max_logit_error.toExponential(2)} over 16 test images.`;
    $('loading').hidden=true;updateMetrics();learningCurve();await selectCase(results.cases[0]);
    // Deliberate test hook: public exported model and identical live preprocessing only.
    window.pneumoniaLab={results,model,forward,loadImage};
    try{const r=await fetch('assets/project.json');if(r.ok){const meta=await r.json();if(meta.repository_url)$('github-link').href=meta.repository_url;}}catch{}
  }catch(e){$('loading').classList.add('error-message');$('loading').textContent=`The experiment could not load: ${e.message}. Please refresh or try again later.`;}
}
document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{filter=b.dataset.filter;document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('active',x===b));drawGallery();});
$('reveal').onclick=()=>{if(output)showTruth();};
for(const kind of ['normal','pneumonia'])$(`guess-${kind}`).onclick=()=>{document.querySelectorAll('.guess-row button').forEach(b=>b.classList.remove('selected'));$(`guess-${kind}`).classList.add('selected');showTruth();};
$('threshold').oninput=e=>{threshold=+e.target.value;updateMetrics();};
$('reset-threshold').onclick=()=>{threshold=run.threshold;$('threshold').value=threshold;updateMetrics();};
$('close-inspector').onclick=()=>$('inspector').close();
$('inspector').onclick=e=>{if(e.target===$('inspector')){const r=$('inspector').getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)$('inspector').close();}};
$('channel-select').onchange=e=>{inspectChannel=+e.target.value;refreshInspector();};
$('input-channel-select').onchange=e=>{inputChannel=+e.target.value;refreshInspector();};
$('inspect-map').onclick=e=>{const r=e.target.getBoundingClientRect();pixel={x:Math.min(inspectLayer.w-1,Math.floor((e.clientX-r.left)/r.width*inspectLayer.w)),y:Math.min(inspectLayer.h-1,Math.floor((e.clientY-r.top)/r.height*inspectLayer.h))};refreshInspector();};
$('inspect-map').onkeydown=e=>{const moves={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]};if(moves[e.key]){e.preventDefault();pixel={x:Math.max(0,Math.min(inspectLayer.w-1,pixel.x+moves[e.key][0])),y:Math.max(0,Math.min(inspectLayer.h-1,pixel.y+moves[e.key][1]))};refreshInspector();}};
init();
