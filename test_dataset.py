from datasets import Dataset

ds = Dataset("initslurtest", annotation_format="onsets")
print(ds.get_annotation_paths())