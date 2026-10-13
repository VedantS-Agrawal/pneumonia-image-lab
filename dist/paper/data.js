// Exact table values from arXiv:2505.06646v1. No raw predictions reconstructed.
export const source='https://arxiv.org/html/2505.06646v1';
export const models=[
 {id:'danny',name:'DannyNet',auc:.8527,f1:.3861,loss:.0416},
 {id:'vit',name:'ViT',auc:.7940,f1:.1114,loss:.1589},
 {id:'replica',name:'CheXNet replica',auc:.7928,f1:.0763,loss:.1661}
];
// Table 2: original CheXNet, DannyNet, ViT, replica.
export const diseases=[
 ['Atelectasis',.809,.817,.774,.762],['Cardiomegaly',.925,.932,.890,.922],
 ['Consolidation',.790,.783,.789,.746],['Edema',.888,.896,.876,.864],
 ['Effusion',.864,.905,.857,.883],['Emphysema',.937,.963,.828,.850],
 ['Fibrosis',.805,.814,.772,.766],['Hernia',.916,.997,.872,.925],
 ['Infiltration',.735,.708,.700,.673],['Mass',.868,.919,.783,.824],
 ['Nodule',.780,.789,.673,.646],['Pleural thickening',.806,.801,.766,.756],
 ['Pneumonia',.768,.740,.713,.656],['Pneumothorax',.889,.875,.821,.827]
];
// Table 4: DannyNet, ViT, replica. Pneumothorax is absent from the source.
export const diseaseF1=[
 [.421,.127,.026],[.532,.264,.423],[.226,0,0],[.286,.004,0],
 [.623,.427,.459],[.516,.079,0],[.127,0,0],[.750,0,0],
 [.395,.193,.061],[.477,.213,.079],[.352,.041,0],[.258,0,0],
 [.082,0,0],[null,null,null]
];
export function classificationMetrics(labels,scores,t){
 let tp=0,tn=0,fp=0,fn=0;
 labels.forEach((y,i)=>{if(y){if(scores[i]>=t)tp++;else fn++;}else{if(scores[i]>=t)fp++;else tn++;}});
 const precision=tp+fp?tp/(tp+fp):0,recall=tp+fn?tp/(tp+fn):0;
 return {tp,tn,fp,fn,precision,recall,f1:2*tp+fp+fn?2*tp/(2*tp+fp+fn):0,accuracy:(tp+tn)/(labels.length||1)};
}
export function auc(labels,scores){
 let wins=0,pairs=0;
 for(let i=0;i<labels.length;i++)if(labels[i])for(let j=0;j<labels.length;j++)if(!labels[j]){wins+=scores[i]>scores[j]?1:scores[i]===scores[j]?.5:0;pairs++;}
 return pairs?wins/pairs:null;
}
export function focal(pt,gamma){return -Math.pow(1-pt,gamma)*Math.log(pt);}
// Teaching examples deliberately invented, never claimed as paper predictions.
export const toyLabels=Array.from({length:100},(_,i)=>i<10?1:0);
export const toyScores=Array.from({length:100},(_,i)=>i<10?.18+i*.024:.015+((i*37)%90)/90*.285);
export const perClassToys=[
 {name:'Finding A',labels:[1,1,1,1,0,0,0,0,0,0],scores:[.91,.83,.69,.62,.56,.41,.35,.21,.12,.05]},
 {name:'Finding B',labels:[1,1,0,0,0,0,0,0,0,0],scores:[.41,.38,.32,.27,.22,.17,.12,.09,.05,.03]},
 {name:'Finding C',labels:[1,0,0,0,0,0,0,0,0,0],scores:[.19,.14,.11,.10,.08,.06,.05,.04,.02,.01]}
];
