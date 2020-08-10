Speech Activity Classifier
=========
Unsupervised solution for Kaggle InClass competition: https://www.kaggle.com/c/silero-audio-classifier/
Extracts and stores as .csv 224 features from .wav audio files and clusters them into 3 classes, corresponding to speech, music and noise.
List of features consists of:
	- means and stds of 11 mfcc 
	- brightness (amount of relative energy corresponds to the frequencies higher than the threshold)
	- roloff (frequency index before which 85% of the energy was accumulated)
	- 1, 25, 50, 75, 99 percentiles for 40 mfccs

Some features were inspired by the project https://github.com/victorwegeborn/Audio-Signal-Feature-Extraction-And-Clustering.

First clusters are with by BayesianGaussianMixture, as it's a more flexible method than Kmeans. 
Then clusters are relabeled using manually obtained mapping (in order to exclude data leak).

Usage
=============

Make a submisson for the competition
------------
```
python clustering.py
```

Inference on wav files from the audio folder
------------
```
python clustering.py -t inference -p <audio folder>
```

Results
=============
Model accuracy according to the leaderboard: 0.832
for model trained only on audiofiles from train/0, train/1, totally 34268 files (~10% of the dataset).

<img align="center" width="420" height="315" src="https://github.com/Evrin-aus-dem-Norden/Speech-Activity-Classifier/blob/master/clusters.png">

Clusters visualization was obtained as follows: highly correlated (> 0.5) with predictions features from train were transformed to 2D space with PCA.