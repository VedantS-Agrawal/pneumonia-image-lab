"""Run: python -m src.train. Test results never select training settings."""
import argparse
import copy
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import medmnist
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from sklearn.metrics import (
    average_precision_score, confusion_matrix, roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

from .model import TinyPneumoniaCNN, image_tensor

ROOT = Path(__file__).resolve().parents[1]


def scores(model, x, device, batch=128):
    model.eval()
    out = []
    with torch.no_grad():
        for chunk in x.split(batch):
            out.append(torch.sigmoid(model(chunk.to(device))).cpu().numpy())
    return np.concatenate(out)


def threshold_from_validation(y, p):
    fpr, tpr, thresholds = roc_curve(y, p)
    finite = np.isfinite(thresholds)
    thresholds = thresholds[finite]
    return float(thresholds[np.argmax((tpr-fpr)[finite])])


def metrics(y, p, threshold):
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    sensitivity = tp / max(1, tp+fn)
    specificity = tn / max(1, tn+fp)
    return dict(auc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)),
                sensitivity=float(sensitivity), specificity=float(specificity),
                accuracy=float((tp+tn)/len(y)), balanced_accuracy=float((sensitivity+specificity)/2),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def auc_interval(y, p):
    rng = np.random.default_rng(2026)
    estimates = []
    for _ in range(1000):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            estimates.append(roc_auc_score(y[idx], p[idx]))
    return np.quantile(estimates, [.025, .975]).tolist()


def train_one(config, size, seed, arrays, device):
    run = ROOT / 'runs' / f'{size}-{seed}'
    run.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = TinyPneumoniaCNN().to(device)
    x, y = arrays['train']
    vx, vy = arrays['val']
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(x, torch.from_numpy(y).float()),
                        batch_size=config['batch_size'], shuffle=True, generator=g, num_workers=0)
    ratio = float((y == 0).sum() / (y == 1).sum())
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ratio, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    best_auc, best_state, stale = -1, None, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, config['epochs']+1):
        tick = time.perf_counter()
        model.train()
        total = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()
            total += loss.item()*len(by)
        vp = scores(model, vx, device)
        auc = float(roc_auc_score(vy, vp))
        row = dict(epoch=epoch, loss=total/len(y), val_auc=auc, seconds=time.perf_counter()-tick)
        history.append(row)
        print(f'size={size} seed={seed} epoch={epoch} loss={row["loss"]:.4f} val_auc={auc:.4f} seconds={row["seconds"]:.1f}', flush=True)
        if auc > best_auc:
            best_auc, stale = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config['patience']:
            break
    model.load_state_dict(best_state)
    vp = scores(model, vx, device)
    threshold = threshold_from_validation(vy, vp)
    tx, ty = arrays['test']
    tp = scores(model, tx, device)
    result = dict(size=size, seed=seed, parameters=sum(p.numel() for p in model.parameters()),
                  threshold=threshold, best_val_auc=best_auc, best_epoch=max(history, key=lambda r:r['val_auc'])['epoch'],
                  training_seconds=time.perf_counter()-started, history=history,
                  validation=metrics(vy,vp,threshold), test=metrics(ty,tp,threshold),
                  test_auc_ci=auc_interval(ty,tp))
    model = model.cpu()
    torch.save(best_state, run/'model.pt')
    np.savez(run/'predictions.npz', val=vp, test=tp)
    (run/'result.json').write_text(json.dumps(result, indent=2))
    return model, result, tp


def export_model(model, size, run_result, x):
    assets = ROOT/'dist/assets'
    weights = {name:dict(shape=list(value.shape), values=value.flatten().tolist())
               for name,value in model.state_dict().items()}
    (assets/f'model-{size}.json').write_text(json.dumps(dict(size=size, threshold=run_result['threshold'],weights=weights),separators=(',',':')))
    example = x[:1]
    model.eval()
    torch.onnx.export(model, example, str(assets/f'model-{size}.onnx'), dynamo=False,
                      input_names=['image'], output_names=['logit'], opset_version=17)
    session = ort.InferenceSession(str(assets/f'model-{size}.onnx'), providers=['CPUExecutionProvider'])
    with torch.no_grad(): expected=model(x[:16]).numpy()
    actual = np.array([session.run(None, {'image':row[None].numpy()})[0][0] for row in x[:16]])
    error=float(np.max(np.abs(expected-actual)))
    assert error < 1e-4, f'ONNX parity error {error}'
    timings=[]
    with torch.no_grad():
        for _ in range(5):model(example)
        for _ in range(50):
            tick=time.perf_counter();model(example);timings.append((time.perf_counter()-tick)*1000)
    run_result['cpu_single_image_ms']=float(np.median(timings))
    run_result['onnx_max_logit_error']=error


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(ROOT/'configs/experiment.json'))
    parser.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    torch.set_num_threads(4)
    device='mps' if args.device=='auto' and torch.backends.mps.is_available() else ('cpu' if args.device=='auto' else args.device)
    print('Training device:',device,flush=True)
    (ROOT/'data').mkdir(exist_ok=True)
    (ROOT/'dist/assets/samples').mkdir(exist_ok=True)
    results=[]; predictions={}; images={}; reference_y=None; splits={}
    for size in config['sizes']:
        arrays={}
        for split in ['train','val','test']:
            ds=medmnist.PneumoniaMNIST(split=split,size=size,download=True,root=str(ROOT/'data'))
            source_images = ds.imgs
            y=ds.labels.flatten().astype(int)
            arrays[split]=(image_tensor(source_images),y)
            if split=='test':
                images[size]=source_images
                if reference_y is not None:assert np.array_equal(y,reference_y)
                reference_y=y
            splits[split]=dict(n=len(y),normal=int((y==0).sum()),pneumonia=int((y==1).sum()))
        for seed in config['seeds']:
            model,result,p=train_one(config,size,seed,arrays,device)
            if seed==config['representative_seed']:
                export_model(model,size,result,arrays['test'][0])
                predictions[size]=p
            results.append(result)
    # Gallery: fixed random cases plus error examples. Report selection, never use for metrics.
    rng=np.random.default_rng(17)
    ids=set(rng.choice(len(reference_y),24,replace=False).tolist())
    for size,p in predictions.items():
        threshold=next(r['threshold'] for r in results if r['size']==size and r['seed']==config['representative_seed'])
        wrong=np.flatnonzero((p>=threshold)!=reference_y)
        rank=sorted(wrong.tolist(),key=lambda i:abs(float(p[i])-threshold),reverse=True)
        ids.update(rank[:8])
    cases=[]
    for i in sorted(ids):
        for size in config['sizes']:
            Image.fromarray(images[size][i]).save(ROOT/f'dist/assets/samples/{i}-{size}.png')
        cases.append(dict(id=i,label=int(reference_y[i]),scores={str(size):float(predictions[size][i]) for size in config['sizes']}))
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),dataset='PneumoniaMNIST',config=config,
                  preprocessing='Official 28×28 grayscale images, no additional resizing; uint8 normalized to [-1,1].',
                  device=device,hardware=platform.machine(),platform=platform.platform(),torch_version=torch.__version__,
                  split_counts=splits,runs=results,cases=cases,
                  test_labels=reference_y.tolist(),test_scores={str(k):v.tolist() for k,v in predictions.items()},
                  selection='24 random test images (seed 17), plus up to 8 most confidently incorrect examples per representative model; gallery is not representative.',
                  bootstrap='1,000 image-level bootstrap samples, seed 2026; does not account for patient clustering or external generalization.',
                  interpretation='Model scores are not calibrated clinical risks; weighted loss changes score interpretation.')
    (ROOT/'dist/assets/results.json').write_text(json.dumps(manifest,separators=(',',':')))
    # Small, exact fixture for browser/Python inference validation.
    fixture=dict()
    for size,p in predictions.items():
        ids3=[0,1,2]
        for i in ids3:Image.fromarray(images[size][i]).save(ROOT/f'dist/assets/samples/{i}-{size}.png')
        fixture[str(size)]=[dict(id=i,score=float(p[i])) for i in ids3]
    (ROOT/'dist/assets/parity.json').write_text(json.dumps(fixture))
    summary={str(s):[r['test']['auc'] for r in results if r['size']==s] for s in config['sizes']}
    print('Finished. Test AUC by resolution:',summary,flush=True)


if __name__=='__main__':main()
