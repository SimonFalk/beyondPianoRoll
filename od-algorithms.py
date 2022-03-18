import numpy as np
from madmom.features.onsets import OnsetPeakPickingProcessor, CNNOnsetProcessor, SpectralOnsetProcessor, peak_picking

proc = OnsetPeakPickingProcessor(threshold=0.1)
sodf = SpectralOnsetProcessor(onset_method="high_frequency_content")
act_fn = sodf('../DoReMir/initslurtest_vn/slurtest-all.wav')

onset_idx = proc(act_fn)

f = open("../DoReMir/initslurtest_vn/hfc015_onsets_all-madmom.csv", "w")
for i in range(len(onset_idx)):
    f.write(str(np.round(onset_idx[i]*0.01, decimals=2))+ "\n")
f.close()
