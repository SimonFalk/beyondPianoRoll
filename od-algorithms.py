import numpy as np
from madmom.features.onsets import OnsetPeakPickingProcessor, CNNOnsetProcessor, SpectralOnsetProcessor, peak_picking
import matplotlib.pyplot as plt

proc = OnsetPeakPickingProcessor(threshold=0.1)
sodf = CNNOnsetProcessor()
act_fn = sodf("datasets/OnsetLabeledInstr2013/development/Violin/42954_FreqMan_hoochie_violin_pt1.wav")

onset_idx = proc(act_fn)

plt.plot(act_fn)
plt.show()

'''
f = open("../DoReMir/initslurtest_vn/hfc015_onsets_all-madmom.csv", "w")
for i in range(len(onset_idx)):
    f.write(str(np.round(onset_idx[i]*0.01, decimals=2))+ "\n")
f.close()
'''