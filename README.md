Speech Activity Classifier
=========
Unsupervised solution for Kaggle InClass competition: https://www.kaggle.com/c/silero-audio-classifier/

Extracts and stores as .csv 224 features from .wav audio files and clusters them into 3 classes, corresponding to speech, music and noise.
Some features were inspired by the project https://github.com/victorwegeborn/Audio-Signal-Feature-Extraction-And-Clustering.
For training were used only audiofiles from train/0, train/1, totally 34268 files (~10% of the dataset).

Model accuracy according to the leaderboard: 0.832