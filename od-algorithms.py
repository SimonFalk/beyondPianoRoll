import numpy as np
from madmom.features.onsets import CNNOnsetProcessor
import matplotlib.pyplot as plt
from madmom.ml.nn import NeuralNetwork

#proc = OnsetPeakPickingProcessor(threshold=0.1)
proc = CNNOnsetProcessor()
res = proc("datasets/OnsetLabeledInstr2013/development/Violin/42954_FreqMan_hoochie_violin_pt1.wav")

debug = 0
#onset_idx = proc(act_fn)


'''
np.save(
    "results/madmomCNNOnsetProcessor/42954_FreqMan_hoochie_violin_pt1", 
    act_fn
)

f = open("../DoReMir/initslurtest_vn/hfc015_onsets_all-madmom.csv", "w")
for i in range(len(onset_idx)):
    f.write(str(np.round(onset_idx[i]*0.01, decimals=2))+ "\n")
f.close()
'''