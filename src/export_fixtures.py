"""Export raw-pixel fixtures after training, for JavaScript parity verification."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    x=np.load(ROOT/'data/pneumoniamnist.npz')['test_images']
    r=json.loads((ROOT/'dist/assets/results.json').read_text())
    ids=[0,1,2,10,100,200,300,400,500,623]
    fixtures=[dict(id=i,pixels=x[i].flatten().tolist(),score=r['test_scores']['28'][i]) for i in ids]
    (ROOT/'tests/parity-fixtures.json').write_text(json.dumps(fixtures))
if __name__=='__main__':main()
