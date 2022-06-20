import os
import glob
source_dir = "../DoReMir/articulation"

def get_wav_paths(num_per_dir=10000):
    out_dict = {}
    curr_folder = None
    for fname in os.listdir(source_dir):
        wav_files = []
        p = os.path.join(source_dir, fname)
        if os.path.isdir(p):
            curr_folder = fname

        i = 0
        for sub_fname in glob.glob(os.path.join(p,'*.wav')):
            if sub_fname not in ["../DoReMir/articulation/C_whistle/c536.wav"] and i<num_per_dir:
                wav_files.append(sub_fname)
            i+=1
        
        out_dict[curr_folder] = wav_files
    
    return out_dict
