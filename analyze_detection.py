import sys
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

def f_score(evaluation_result):
	CD,FN,FP,doubles,merged = evaluation_result
	return 2*CD/(2*CD+FP+FN)

def evaluate(true, pred, tol_sec):

	pred_counted = np.zeros(len(pred))
	CD = 0
	FN = 0
	FP = 0
	doubles = 0

	for t in range(len(true)):
		matches = np.where(np.abs(pred-true[t])<tol_sec, 1, 0)
		
		#print("True onset at time {}, found {} match(es)".format(true_events[t], np.sum(matches)))
		if np.sum(matches) == 1:
			# Correct detection
			CD += 1
		elif np.sum(matches) == 0:
			# False negative
			FN += 1
		elif np.sum(matches)>1:
			doubles += 1
			FP += int(np.sum(matches)-1)
			CD += 1

		pred_counted = pred_counted + matches
	
	FP += np.count_nonzero(pred_counted<0.001)
	merged = np.count_nonzero(pred_counted>1)
	return (CD,FN,FP,doubles,merged)

# Preprocess

'''
proc_tol = 0.03
proc_events = []
curr = 0
for i in range(len(pred_events)):
	if pred_events[i]-curr > proc_tol:
		proc_events.append(pred_events[i])
		curr = pred_events[i]

pred_events = np.array(proc_events)
'''

'''
if input("Save processed onsets?")=="y":
	f = open(sys.argv[2][:-4] + "-proc.csv", "w")
	f.write("\n".join([str(ev) for ev in proc_events]))
	f.close()

'''
if __name__=="__main__":
    true_events = genfromtxt(sys.argv[1], delimiter=',')
    pred_events = genfromtxt(sys.argv[2], delimiter=',')
    
    tolerance = 0.030
    [CD,FN,FP,doubles,merged] = evaluate(true_events, pred_events, tolerance)

    print("CD:\t{}\tFN:\t{}\tFP:\t{}\tDo:\t{}\tMe:\t{}".format(CD,FN,FP,doubles,merged))