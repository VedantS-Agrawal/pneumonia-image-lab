import unittest
import numpy as np
import torch
from src.model import TinyPneumoniaCNN,image_tensor
from src.train import metrics,threshold_from_validation
class Contracts(unittest.TestCase):
    def test_input_preprocessing(self):
        x=image_tensor(np.array([[[0,255]]],dtype=np.uint8))
        self.assertEqual(tuple(x.shape),(1,1,1,2))
        self.assertEqual(x.flatten().tolist(),[-1.,1.])
    def test_architecture(self):
        m=TinyPneumoniaCNN()
        self.assertEqual(sum(p.numel() for p in m.parameters()),4753)
        self.assertEqual(tuple(m(torch.zeros(2,1,28,28)).shape),(2,))
    def test_threshold_and_metrics(self):
        y=np.array([0,0,1,1]);p=np.array([.1,.6,.4,.9])
        t=threshold_from_validation(y,p)
        self.assertTrue(np.isfinite(t))
        r=metrics(y,p,.5)
        self.assertEqual((r['tn'],r['fp'],r['fn'],r['tp']),(1,1,1,1))
        self.assertEqual(r['balanced_accuracy'],.5)
if __name__=='__main__':unittest.main()
