import {models,diseases,diseaseF1,classificationMetrics,auc,focal,toyLabels,toyScores,perClassToys} from './data.js';
const $=id=>document.getElementById(id),pct=v=>(v*100).toFixed(1)+'%';
let metric='auc';
function bars(target,rows,max=1,digits=4){
 $(target).replaceChildren();
 rows.forEach(([name,v,color])=>{const row=document.createElement('div');row.className='bar-row';
 const label=document.createElement('span');label.textContent=name;
 const track=document.createElement('div');track.className='bar-track';const bar=document.createElement('div');bar.className='bar-fill';bar.style.width=v===null?'0%':pct(v/max);if(color)bar.style.background=color;track.append(bar);
 const num=document.createElement('strong');num.textContent=v===null?'—':v.toFixed(digits);if(v===null)num.title='Not reported in this source table';row.append(label,track,num);$(target).append(row);});
}
function metricBars(){
 bars('model-bars',models.map(m=>[m.name,m[metric]]),metric==='loss'?.2:1);
 const explanation={auc:'Mean AUC ranks examples separately for each label, then averages. A score of 0.85 is not 85% accuracy or sensitivity.',f1:'F1 depends on decisions at a threshold. DannyNet alone uses per-class thresholds; the other two use 0.5 (paper evaluation section).',loss:'Loss measures an objective. DannyNet uses focal loss while the baselines use BCE; these numbers should not be treated as a common scale.'};
 $('metric-explanation').textContent=explanation[metric];
}
function diseaseBars(){
 const i=+$('disease-select').value,row=diseases[i];
 $('disease-source').textContent=$('disease-metric').value==='f1'?'Reported · Table 4':'Reported · Table 2';
 if($('disease-metric').value==='f1'){
  const values=diseaseF1[i];
  bars('disease-bars',[['Original CheXNet',null,'#aeb9c7'],['DannyNet',values[0],'#365ee7'],['ViT',values[1],'#8799c7'],['Replica',values[2],'#6c8495']],1,3);
  $('disease-insight').textContent=values[0]===null?`${row[0]}: Table 4 omits this finding. Dashes mean missing values, not zero performance.`:`${row[0]}: DannyNet F1 ${values[0].toFixed(3)}, ViT ${values[1].toFixed(3)}, replica ${values[2].toFixed(3)}. Only DannyNet uses class-specific thresholds; this comparison includes a threshold-policy difference. Original CheXNet F1 is not supplied in Table 4.`;
  return;
 }
 bars('disease-bars',[['Original CheXNet',row[1],'#aeb9c7'],['DannyNet',row[2],'#365ee7'],['ViT',row[3],'#8799c7'],['Replica',row[4],'#6c8495']],1,3);
 const d=row[2]-row[1],r=row[2]-row[4];
 $('disease-insight').textContent=`${row[0]}: DannyNet is ${d>=0?'+':''}${d.toFixed(3)} versus the original-study value and ${r>=0?'+':''}${r.toFixed(3)} versus the replica. ${d<0?'This row is a counterexample to uniform improvement.':'This row reports an improvement; uncertainty and evaluation comparability still matter.'}`;
}
function initLabels(){
 diseases.forEach(([name],i)=>{const l=document.createElement('label');const c=document.createElement('input');c.type='checkbox';c.value=i;c.onchange=labelVector;l.append(c,document.createTextNode(name));$('label-toggles').append(l);});labelVector();
}
function labelVector(){const vector=Array.from($('label-toggles').querySelectorAll('input'),c=>c.checked?1:0);$('label-vector').textContent=`[${vector.join(', ')}] · ${vector.reduce((a,b)=>a+b,0)} positive labels · 2¹⁴ = 16,384 possible vectors`;}
function split(mode){
 const groups=mode==='patient'?[[0,0,1,1,2,2,3,3],[4,4,5,5]]:[[0,1,2,3,4,5],[0,1,2,3,4,5]];
 $('patient-split').replaceChildren();
 groups.forEach((ids,i)=>{const group=document.createElement('div');group.className='split-group';const h=document.createElement('h4');h.textContent=i?'Test images':'Training images';group.append(h);const seen={};ids.forEach(p=>{seen[p]=(seen[p]||0)+1;const c=document.createElement('span');c.className='patient-chip'+(mode==='image'?' leak':'');c.textContent=`Patient ${String.fromCharCode(65+p)} · image ${mode==='image'?i+1:seen[p]}`;group.append(c);});$('patient-split').append(group);});
 $('leakage-result').className='notice '+(mode==='image'?'warning':'success');$('leakage-result').textContent=mode==='image'?'All six patients appear on both sides. Related images can leak patient-specific information. This illustration does not quantify performance inflation.':'No patient appears on both sides. Evaluate patients the model has not encountered; retain a separate validation group in a real study.';
}
function dense(enabled){
 const n=5,x=i=>45+i*100,svg=[];
 for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)if(enabled||j===i+1){const y=85,r=(j-i)*15;svg.push(`<path d="M${x(i)} 85 Q${(x(i)+x(j))/2} ${85-r} ${x(j)} 85" fill="none" stroke="${j===i+1?'#365ee7':'#a7b5d8'}" stroke-width="${j===i+1?2:1}"/>`);}
 for(let i=0;i<n;i++)svg.push(`<rect x="${x(i)-25}" y="70" width="50" height="35" rx="7" fill="#e8edff" stroke="#9aafe9"/><text x="${x(i)}" y="91" text-anchor="middle" font-size="12" fill="#2a4260">L${i}</text>`);
 $('dense-diagram').innerHTML=svg.join('');$('dense-explanation').textContent=enabled?'Later layers receive earlier feature maps through concatenation. This supports feature reuse; a connecting line is not a new disease prediction.':'In this schematic, each layer receives only the previous layer’s features. Actual models can have additional connection patterns.';
}
function loss(){
 const pt=+$('pt').value,g=+$('gamma').value,bce=-Math.log(pt),fl=focal(pt,g);
 $('pt-value').textContent=pt.toFixed(2);$('gamma-value').textContent=g.toFixed(2);$('bce-value').textContent=bce.toFixed(4);$('focal-value').textContent=fl.toFixed(4);
 const curve=fn=>Array.from({length:99},(_,i)=>{const x=(i+1)/100;return `${25+x*390},${135-fn(x)/5*110}`;}).join(' ');
 $('loss-curve').innerHTML=`<line x1="25" y1="135" x2="420" y2="135" stroke="#d5dfe3"/><polyline points="${curve(p=>-Math.log(p))}" fill="none" stroke="#8198b2" stroke-width="2"/><polyline points="${curve(p=>focal(p,g))}" fill="none" stroke="#365ee7" stroke-width="2"/><circle cx="${25+pt*390}" cy="${135-fl/5*110}" r="4" fill="#365ee7"/><text x="270" y="20" font-size="10" fill="#8198b2">BCE</text><text x="320" y="20" font-size="10" fill="#365ee7">Focal loss</text><text x="25" y="155" font-size="9" fill="#70828c">pₜ = 0</text><text x="370" y="155" font-size="9" fill="#70828c">pₜ = 1</text>`;
}
function toy(){
 const t=+$('toy-threshold').value,m=classificationMetrics(toyLabels,toyScores,t);
 $('toy-threshold-value').textContent=t.toFixed(2);$('toy-auc').textContent=auc(toyLabels,toyScores).toFixed(3);$('toy-precision').textContent=pct(m.precision);$('toy-recall').textContent=pct(m.recall);$('toy-f1').textContent=m.f1.toFixed(3);
 $('score-dots').replaceChildren();
 toyScores.forEach((s,i)=>{const d=document.createElement('span');d.className='score-dot'+(toyLabels[i]?' positive':'');d.style.left=`calc(${pct(s)} - 5px)`;d.style.top=(toyLabels[i]?15+i*10:20+(i%10)*11)+'px';d.title=`Synthetic ${toyLabels[i]?'positive':'negative'} · score ${s.toFixed(3)}`;$('score-dots').append(d);});
 const line=document.createElement('span');line.className='threshold-line';line.style.left=pct(t);$('score-dots').append(line);
 $('toy-counts').textContent=`TP ${m.tp} · FP ${m.fp} · FN ${m.fn} · TN ${m.tn}. ${t>.42?'All positive scores are below this threshold, so F1 is zero despite a useful ranking.':'Changing the threshold changes decisions, not the order of scores or AUC.'}`;
}
function classes(custom){
 const thresholds=custom?[.60,.35,.16]:[.5,.5,.5];let sum=0;$('class-f1').replaceChildren();
 perClassToys.forEach((c,i)=>{const m=classificationMetrics(c.labels,c.scores,thresholds[i]);sum+=m.f1;const row=document.createElement('div');row.className='class-row';row.innerHTML=`<span>${c.name} · threshold ${thresholds[i].toFixed(2)}</span><strong>F1 ${m.f1.toFixed(3)}</strong>`;$('class-f1').append(row);});
 $('macro-f1').textContent=`Macro F1 = average of the three class F1 values = ${(sum/3).toFixed(3)}. The scores and underlying ranking have not changed.`;
}
function prevalence(){
 const v=+$('prevalence').value/100,p=1000*v,tp=.9*p,fn=.1*p,fp=.1*(1000-p),tn=.9*(1000-p);
 $('prevalence-value').textContent=pct(v);$('prevalence-result').textContent=`Expected counts: TP ${tp.toFixed(1)}, FN ${fn.toFixed(1)}, FP ${fp.toFixed(1)}, TN ${tn.toFixed(1)}. Precision = ${pct(tp/(tp+fp))}. Even with fixed 90% sensitivity/specificity, many positives can be false alarms when the finding is rare.`;
}
function attribution(type){
 $('attribution-grid').replaceChildren();
 for(let y=0;y<8;y++)for(let x=0;x<8;x++){const cx=type===0?2:5,cy=type===0?5:2,v=Math.exp(-((x-cx)**2+(y-cy)**2)/5);const d=document.createElement('div');d.style.background=`rgb(${Math.round(235+v*15)},${Math.round(240-v*135)},${Math.round(245-v*150)})`;$('attribution-grid').append(d);}
}
function calibration(){
 const n=+$('calibration').value;$('calibration-value').textContent=n;
 $('calibration-result').textContent=`Assigned score: 0.80. Observed positive fraction: ${(n/100).toFixed(2)}. ${n===80?'These hypothetical observations agree with the assigned score in this bin.':'The score and observed frequency disagree in this hypothetical bin.'} One bin alone does not establish overall calibration.`;
}
function quiz(){
 const items=[
 ['An AUC of 0.85 means…',['85% of images are correctly classified','Positives tend to rank above negatives','Sensitivity is always 85%'],1,'AUC describes ranking across thresholds. Accuracy and sensitivity require a chosen operating point.'],
 ['What would make the F1 comparison fairer?',['Tune all model thresholds on validation using the same procedure','Choose every threshold on the test set','Give only the best model a lower threshold'],0,'Apply a consistent validation-based procedure, then compare on an untouched test set.'],
 ['Does lower focal loss prove better calibration?',['Yes, lower is always more calibrated','No; objectives and calibration measurements differ'],1,'Use reliability diagrams or appropriate calibration metrics. BCE and focal-loss values are not interchangeable.'],
 ['What does a bright Grad-CAM region establish?',['The disease is definitely present there','The model attribution emphasizes that region','The segmentation is correct'],1,'Attribution is class- and layer-dependent. Clinical correctness requires separate evidence.'],
 ['Can 0.952 in the small lab be ranked against 0.8527 here?',['Yes; higher AUC wins','No; tasks and evaluation data differ'],1,'A binary pediatric benchmark and a 14-label benchmark are different experiments.']
 ];
 items.forEach(([q,answers,correct,explanation],i)=>{const item=document.createElement('div');item.className='quiz-item';const h=document.createElement('h4');h.textContent=`${i+1}. ${q}`;const options=document.createElement('div');options.className='quiz-options';const feedback=document.createElement('div');feedback.className='quiz-feedback';feedback.setAttribute('aria-live','polite');answers.forEach((a,j)=>{const b=document.createElement('button');b.textContent=a;b.onclick=()=>{options.querySelectorAll('button').forEach(x=>x.className='');b.className=j===correct?'correct':'incorrect';feedback.textContent=(j===correct?'Correct. ':'Reconsider. ')+explanation;};options.append(b);});item.append(h,options,feedback);$('quiz').append(item);});
}
function activate(group,button){document.querySelectorAll(group).forEach(b=>b.classList.toggle('active',b===button));}
document.querySelectorAll('[data-metric]').forEach(b=>b.onclick=()=>{metric=b.dataset.metric;activate('[data-metric]',b);metricBars();});
diseases.forEach(([name],i)=>{const o=document.createElement('option');o.value=i;o.textContent=name;$('disease-select').append(o);});$('disease-select').value=12;$('disease-select').onchange=diseaseBars;$('disease-metric').onchange=diseaseBars;
$('auc-table').innerHTML=diseases.map(row=>`<tr><td>${row[0]}</td>${row.slice(1).map(v=>`<td>${v.toFixed(3)}</td>`).join('')}</tr>`).join('');
document.querySelectorAll('[data-split]').forEach(b=>b.onclick=()=>{activate('[data-split]',b);split(b.dataset.split);});
$('sequential-mode').onclick=e=>{activate('#sequential-mode,#dense-mode',e.target);dense(false);};$('dense-mode').onclick=e=>{activate('#sequential-mode,#dense-mode',e.target);dense(true);};
$('jitter').oninput=e=>{$('jitter-image').style.filter=`brightness(${e.target.value})`;$('jitter-value').textContent=(+e.target.value).toFixed(2)+'× brightness';};
$('pt').oninput=loss;$('gamma').oninput=loss;$('toy-threshold').oninput=toy;$('prevalence').oninput=prevalence;$('calibration').oninput=calibration;
$('global-threshold').onclick=e=>{activate('#global-threshold,#class-threshold',e.target);classes(false);};$('class-threshold').onclick=e=>{activate('#global-threshold,#class-threshold',e.target);classes(true);};
document.querySelectorAll('[data-map]').forEach(b=>b.onclick=()=>{activate('[data-map]',b);attribution(+b.dataset.map);});
initLabels();metricBars();diseaseBars();split('patient');dense(true);loss();toy();classes(false);prevalence();attribution(0);calibration();quiz();
