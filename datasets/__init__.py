import os

##Dataset split:

# A, B, and C1 used for validation/development
# C2 used for test
# C2 chosen by sampling 2 random recordings from the "tricky"(first 4) in C
# and sampling 2 random recordings from the "easy"(last 9) in C
# and an additional recording containing vibrato.

PATHS = {
        "holzap_dev" : [
            'datasets/OnsetLabeledInstr2013/development/Piano/piano1', 
            'datasets/OnsetLabeledInstr2013/development/Piano/mussorgsky', 
            'datasets/OnsetLabeledInstr2013/development/Piano/p5', 
            'datasets/OnsetLabeledInstr2013/development/Piano/p3', 
            'datasets/OnsetLabeledInstr2013/development/Piano/MOON', 
            'datasets/OnsetLabeledInstr2013/development/Piano/autumn', 
            'datasets/OnsetLabeledInstr2013/development/Violin/violin2', 
            'datasets/OnsetLabeledInstr2013/development/Violin/my_violin2', 
            'datasets/OnsetLabeledInstr2013/development/Violin/42954_FreqMan_hoochie_violin_pt1', 
            'datasets/OnsetLabeledInstr2013/development/Violin/my_violin1', 
            'datasets/OnsetLabeledInstr2013/development/Violin/my_violin3', 
            'datasets/OnsetLabeledInstr2013/development/Guitar/Summer_Together_110_pt1', 
            'datasets/OnsetLabeledInstr2013/development/Guitar/my_guitar1', 
            'datasets/OnsetLabeledInstr2013/development/Guitar/2684_TexasMusicForge_Dandelion_pt1', 
            'datasets/OnsetLabeledInstr2013/development/Guitar/guitar2', 
            'datasets/OnsetLabeledInstr2013/development/Guitar/guitar3', 
            'datasets/OnsetLabeledInstr2013/development/Oud/1', 
            'datasets/OnsetLabeledInstr2013/development/Oud/Diverse_-_01_-_Taksim_pt1', 
            'datasets/OnsetLabeledInstr2013/development/Oud/rast_taksim1', 
            'datasets/OnsetLabeledInstr2013/development/Oud/ud_taksimleri_-_02_-_huezzam_taksim_pt1', 
            'datasets/OnsetLabeledInstr2013/development/Oud/8'
        ],
        'slurtest_add_1' : [
            'stormhatten_IR2',
            'slurtest02_IR1',
            'slurtest01_IR2',
            'slurtest03_FK1',
            '6xtpsg_220319',
            'slurtest04_IR2',
            'melodyvib_220319',
            'slurtest09_IR2',
            'janissa_IR2',
            '6xtpsg_220306',
            'slurtest01_FK1',
            'slurtest04_FK1',
            'slurtest01_IR1',
            '63an_start_220306',
            'slurtest08_FK1',
            'slurtest03_IR1',
            'stormhatten_IR1',
        ],
        'slurtest_add_2' : [
            "63an_start_220319",
            "hacketi_220319_start",
            "janissa_IR1",
            "slurtest02_IR2",
            "slurtest03_IR2",
            "slurtest04_IR1",
            "slurtest05_IR1",
            "slurtest05_IR2",
            "slurtest09_FK1",
        ],
        'slurtest_test' : [
            "6xtscale_220306", # C2
            "6xtscale_220319", # C2
            "slurtest05_FK", # C2
            "slurtest09_IR1", # C2
        ]
}
    


class Dataset:
    __DATASET_AUDIO_PATHS = {
        "holzap_dev" : "datasets/OnsetLabeledInstr2013/development/",
        "initslurtest" : "datasets/initslurtest_vn/initslurtest_vn_wav/",
        "slurtest_add_1" : "datasets/slurtest_add/slurtest_add_audio/",
        "slurtest_add_2" : "datasets/slurtest_add/slurtest_add_audio/",
        "slurtest_test" :  "datasets/slurtest_add/slurtest_add_audio/",
    }
    __DATASET_ANNOTATION_PATHS = {
        "holzap_dev" : "datasets/OnsetLabeledInstr2013/development/",
        "initslurtest" : "datasets/initslurtest_vn/initslurtest_vn_annotations/",
        "slurtest_add_1" : "datasets/slurtest_add/slurtest_add_annotations/",
        "slurtest_add_2" : "datasets/slurtest_add/new_annotations/",
         "slurtest_test" :   "datasets/slurtest_add/new_annotations/",
    }
    __AUDIO_FORMATS = {
        "holzap_dev" : "wav",
        "initslurtest" : "wav",
        "slurtest_add_1" : "wav",
        "slurtest_add_2" : "wav",
        "slurtest_test" :   "wav"
    }
    __ANNOTATION_FORMATS = {
        "holzap_dev": "onsets",
        "initslurtest" : "txt",
        "slurtest_add_1" : "txt",
        "slurtest_add_2" : "txt",
         "slurtest_test" :   "txt"
    }

    def __init__(self, dataset_name, audio_format="wav", annotation_format="csv"):
        self.dataset_name = dataset_name
        self.audio_base = self.__DATASET_AUDIO_PATHS[dataset_name]
        self.annotation_base = self.__DATASET_ANNOTATION_PATHS[dataset_name]

        #self.audio_paths, self.annotation_paths = self.mine(audio_format, annotation_format)


    def walk(self, base_path, format):
        id_map = {}
        for root, dirs, files in os.walk(base_path, topdown=False):
            for name in files:
                ident, ending = name.split(".")
                if ending == format:
                    id_map[os.path.join(root, ident)] = os.path.join(root, name)
        return id_map

    def mine(self, audio_format, annotation_format):
        audio_paths, annotation_paths = [], []
        audio_map = self.walk(
            self.__DATASET_AUDIO_PATHS[self.dataset_name],
            audio_format
        )
        annotation_map = self.walk(
            self.__DATASET_ANNOTATION_PATHS[self.dataset_name],
            annotation_format
        )
        print("Dataset created")
        print("Found {} audio files and {} annotation files.".format(
            len(audio_map), len(annotation_map)))
        
        self.audio_map = audio_map
        self.annotation_map = annotation_map

        for key in audio_map.keys():
            # Put files in right order
            try:
                annotation_paths.append(annotation_map[key])
            except KeyError:
                continue
            audio_paths.append(audio_map[key])

        return (audio_paths, annotation_paths)

    def get_audio_paths(self):
        if self.dataset_name == "holzap_dev":
            return [p + ".wav" for p in PATHS["holzap_dev"]]
        elif self.dataset_name == "initslurtest":
            base = "datasets/initslurtest_vn/initslurtest_vn_wav/"
            return [base + "slurtest{:02d}.wav".format(i) for i in range(1,20)]
        elif self.dataset_name == 'slurtest_add_1':
            base = "datasets/slurtest_add/slurtest_add_audio/"
            return [base + file + ".wav" for file in PATHS["slurtest_add_1"]]
        elif self.dataset_name == 'slurtest_add_2':
            base = "datasets/slurtest_add/slurtest_add_audio/"
            return [base + file + ".wav" for file in PATHS["slurtest_add_2"]]
        elif self.dataset_name == 'slurtest_test':
            base = "datasets/slurtest_add/slurtest_add_audio/"
            return [base + file + ".wav" for file in PATHS["slurtest_test"]]
        return self.audio_paths
    
    def get_annotation_paths(self):
        if self.dataset_name == "holzap_dev":
            return [p + ".onsets" for p in PATHS["holzap_dev"]]
        elif self.dataset_name == "initslurtest":
            base = "datasets/initslurtest_vn/initslurtest_vn_annotations/"
            return [base + "{:02d}.txt".format(i) for i in range(1,20)]
        elif self.dataset_name == 'slurtest_add_1':
            base = "datasets/slurtest_add/slurtest_add_annotations/"
            return [base + file + ".txt" for file in PATHS["slurtest_add_1"]]
        elif self.dataset_name == 'slurtest_add_2':
            base = "datasets/slurtest_add/new_annotations/"
            return [base + file + ".txt" for file in PATHS["slurtest_add_2"]]
        elif self.dataset_name == 'slurtest_test':
            base = "datasets/slurtest_add/new_annotations/"
            return [base + file + ".txt" for file in PATHS["slurtest_test"]]
        return self.annotation_paths