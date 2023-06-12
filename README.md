# beyondPianoRoll
This is the code repository for the project about *expressive AMT* which will be published in the paper 

Falk, Simon; Sturm, Bob L.T. and Ahlbäck, Sven, "Automatic Legato Transcription based on Onset Detection", in Proceedings of the 20th Sound and Music Computing Conference, June 12-17, 2023.

Copyright Simon Falk 2023

## Description of the repository

- `flow.ipynb` was used to train the model for articulated onset detection.

- `cnn-analysis.ipynb` was used to analyze the performance and training process of the models.

- `data-analysis.ipynb` was used to analyze the training dataset and provide a summary of the data used.

## CNN model for hard onset detection

[To be updated]

For a demo how to apply the CNN HOD model to a test recording and detect onsets, see the
[demo.ipynb](demo.ipynb) notebook.

The trained models are found in the [results](results/) folder.

The path to the optimized model is found [here](model/__init__.py).
