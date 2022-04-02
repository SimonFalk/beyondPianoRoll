import os

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
        ]
}


class Dataset:
    __DATASET_AUDIO_PATHS = {
        "holzap_dev" : "datasets/OnsetLabeledInstr2013/development/",
        "initslurtest" : "datasets/initslurtest_vn/initslurtest_vn_wav/"
    }
    __DATASET_ANNOTATION_PATHS = {
        "holzap_dev" : "datasets/OnsetLabeledInstr2013/development/",
        "initslurtest" : "datasets/initslurtest_vn/initslurtest_vn_annotations/" 
    }
    __AUDIO_FORMATS = {
        "holzap_dev" : "wav",
        "initslurtest" : "wav",
    }
    __ANNOTATION_FORMATS = {
        "holzap_dev": "onsets",
        "initslurtest" : "txt",
    }

    def __init__(self, dataset_name, audio_format="wav", annotation_format="csv"):
        self.dataset_name = dataset_name
        self.audio_base = self.__DATASET_AUDIO_PATHS[dataset_name]
        self.annotation_base = self.__DATASET_ANNOTATION_PATHS[dataset_name]

        self.audio_paths, self.annotation_paths = self.mine(audio_format, annotation_format)


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
        return self.audio_paths
    
    def get_annotation_paths(self):
        if self.dataset_name == "holzap_dev":
            return [p + ".onsets" for p in PATHS["holzap_dev"]]
        elif self.dataset_name == "initslurtest":
            base = "datasets/initslurtest_vn/initslurtest_vn_annotations/"
            return [base + "{:02d}.txt".format(i) for i in range(1,20)]
        return self.annotation_paths