// Educational forward pass. Weights and preprocessing match the PyTorch model.
export function conv(input, weight, bias) {
  const {data,h,w,c}=input, [outC,,kh,kw]=weight.shape;
  const output=new Float32Array(outC*h*w), k=weight.values, b=bias.values;
  for(let oc=0;oc<outC;oc++)for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    let sum=b[oc];
    for(let ic=0;ic<c;ic++)for(let dy=0;dy<kh;dy++){
      const iy=y+dy-1;if(iy<0||iy>=h)continue;
      for(let dx=0;dx<kw;dx++){
        const ix=x+dx-1;if(ix<0||ix>=w)continue;
        sum+=data[ic*h*w+iy*w+ix]*k[((oc*c+ic)*kh+dy)*kw+dx];
      }
    }
    output[oc*h*w+y*w+x]=sum;
  }
  return {data:output,h,w,c:outC};
}
export function relu(a){return {...a,data:Float32Array.from(a.data,x=>Math.max(0,x))};}
export function pool(a){
  const h=Math.floor(a.h/2),w=Math.floor(a.w/2),data=new Float32Array(a.c*h*w);
  for(let c=0;c<a.c;c++)for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    let v=-Infinity;
    for(let dy=0;dy<2;dy++)for(let dx=0;dx<2;dx++)v=Math.max(v,a.data[c*a.h*a.w+(y*2+dy)*a.w+x*2+dx]);
    data[c*h*w+y*w+x]=v;
  }
  return {data,h,w,c:a.c};
}
export function forward(pixels, model){
  const size=model.size, w=model.weights;
  if(pixels.length!==size*size)throw Error('Unexpected input dimensions');
  const layers=[{id:'input',label:'Input',kind:'input',data:Float32Array.from(pixels,x=>x/127.5-1),h:size,w:size,c:1}];
  let a=layers[0];
  for(let n=1;n<=3;n++){
    a={...conv(a,w[`conv${n}.weight`],w[`conv${n}.bias`]),id:`conv${n}`,label:`Conv ${n}`,kind:'conv',parent:layers.at(-1).id,weight:`conv${n}`};layers.push(a);
    a={...relu(a),id:`relu${n}`,label:`ReLU ${n}`,kind:'relu',parent:a.id};layers.push(a);
    if(n<3){a={...pool(a),id:`pool${n}`,label:`Pool ${n}`,kind:'pool',parent:a.id};layers.push(a);}
  }
  const means=new Float32Array(a.c);
  for(let c=0;c<a.c;c++){
    let sum=0;for(let i=0;i<a.h*a.w;i++)sum+=a.data[c*a.h*a.w+i];means[c]=sum/(a.h*a.w);
  }
  let logit=w['head.bias'].values[0];
  const contributions=Array.from(means,(v,i)=>v*w['head.weight'].values[i]);
  for(const v of contributions)logit+=v;
  return {layers,means,contributions,logit,score:1/(1+Math.exp(-logit))};
}
