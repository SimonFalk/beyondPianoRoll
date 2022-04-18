import numpy as np
from madmom.features.onsets import CNNOnsetProcessor
from madmom.features.onsets import OnsetPeakPickingProcessor

import matplotlib.pyplot as plt
from madmom.ml.nn import NeuralNetwork


for t in [0.7, 0.9, 0.95]:
    pp = OnsetPeakPickingProcessor(threshold=t, fps=100)
    proc = CNNOnsetProcessor()
    res = proc("datasets/initslurtest_vn/initslurtest_vn_wav/slurtest02.wav")
    onset_idx = pp(res)



#np.save(
#    "results/madmomCNNOnsetProcessor/42954_FreqMan_hoochie_violin_pt1", 
#    act_fn
#)

f = open("datasets/initslurtest_vn_old/model-output/bockmodel095-slurtest02.txt", "w")
for i in range(len(onset_idx)):
    f.write(str(np.round(onset_idx[i]*0.01, decimals=2))+ "\n")
f.close()
