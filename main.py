import os
import config
import fnmatch
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from relabeled_bayesian_gaussian_mixture import RelabeledBayesianGaussianMixture


def list_audiofiles(path):
    """
    Compose list of paths to all the wav files in a directory tree.
    """
    audiofiles = []

    for dirpath, _, files in os.walk(path):
        for file in fnmatch.filter(files, '*.wav'):
            audiofiles.append(os.path.join(dirpath, file))

    if config.verbose:
        print(f"Found {len(audiofiles)} input audiofiles: ..., {audiofiles[-1]}")

    return audiofiles


def load_data(dataset):
    """
    Load .csv file located in config.prepared_path directory with dataset name.
    """
    loaded = pd.read_csv(os.path.join(config.prepared_path, dataset) + '.csv')
    if config.verbose:
        print(f"Loaded {dataset}: {len(loaded)} x {len(loaded.columns)}")

    return loaded


def extract_eleven_mfcc_stat(wav):
    """
    Calculate means and stds for 11 mfccs with custom args.
    """
    window_width = 0.220
    window_stride = 0.050
    sr = config.sample_rate

    mfcc = librosa.feature.mfcc(wav,
                                sr=sr,
                                n_mfcc=11,
                                dct_type=2,
                                hop_length=int(window_stride * sr),
                                n_fft=int(window_width * sr))

    return np.concatenate((np.mean(mfcc, axis=1), np.std(mfcc, axis=1))).flatten()


def extract_brightness(fft):
    """
    Find how much relative energy corresponds to the frequencies above the threshold.
    """
    frequency_threshold = 1800
    return [np.sum(fft[frequency_threshold:]) / np.sum(fft)]


def extract_rolloff(fft):
    """
    Find the frequency index before which 85% of the energy was accumulated.
    """
    threshold = np.sum(fft) * 0.85
    energy = 0.0

    for i in range(fft.size):
        energy += fft[i]
        if energy >= threshold:
            return [i]


def extract_forty_mfcc_stat(wav):
    """
    Calculate (1, 25, 50, 75, 99) percentiles for 40 mfccs with default args.
    """
    return [np.percentile(row, per) for row in librosa.feature.mfcc(wav, sr=config.sample_rate, n_mfcc=40)
            for per in (1, 25, 50, 75, 99)]


def prepare_data(audiofiles):
    """
    Return DataFrame with features according to the config.features for all audiofiles.
    """
    data = []

    for file in audiofiles:
        wav, _ = sf.read(file)
        fft = np.abs(np.fft.rfft(wav))
        _input = {'eleven_mfcc_stat': wav, 'brightness': fft, 'rolloff': fft, 'forty_mfcc_stat': wav}

        row = []
        for feature in config.features:
            row.extend(globals()[f'extract_{feature}'](_input[feature]))
        data.append(row)

    columns = [feature[0] + str(i) for feature in config.features
               for i in range({'e': 22, 'b': 1, 'r': 1, 'f': 200}[feature[0]])]
    prepared = pd.DataFrame(data, columns=columns)

    if config.verbose:
        print(f"Prepared data: {len(prepared)} x {len(prepared.columns)}")

    return prepared


def save_data(prepared, dataset):
    """
    Save prepared as .csv with name dataset in config.prepared_path.
    """
    prepared.to_csv(os.path.join(config.prepared_path, dataset) + '.csv', index=False)

    if config.verbose:
        print(f"Saved {dataset}.csv")


def handle_wrong_rows(data, dataset):
    """
    For train data just drop rows with NaN, while for test replace NaN rows with previous rows from test.
    """
    if dataset == 'train':
        new_data = data.dropna()
        if len(new_data) == len(data):
            return data

        if config.verbose:
            print(f"Dropped {len(data) - len(new_data)} row(s) from {dataset}")
        return new_data

    if dataset == 'test':
        indexes = set(list(range(len(data)))).difference(set(data.dropna().index))
        if not indexes:
            return data

        for index in sorted(indexes):
            data.iloc[index] = data.iloc[index - 1]

        if config.verbose:
            print(f"Replaced {len(indexes)} row(s) from {dataset}")
        return data


def train_pipeline():
    """
    Load or create dataset, then create and fit pipeline, show its results, and return it trained.
    """
    try:
        train_df = load_data('train')
    except FileNotFoundError:
        train_df = prepare_data(list_audiofiles(config.train_path))
        save_data(train_df, 'train')

    train = np.array(handle_wrong_rows(train_df, 'train'))
    pipeline = Pipeline([('scaler', StandardScaler()),
                         ('clusterization', RelabeledBayesianGaussianMixture(config=config,
                                                                             n_components=config.n_classes,
                                                                             tol=0.00001,
                                                                             covariance_type='tied',
                                                                             max_iter=10000,
                                                                             random_state=18))])

    tr_predictions = pipeline.fit_predict(train)
    targets = pd.read_csv(config.targets).target
    print('\nModel accuracy: %.3f' % (metrics.accuracy_score(targets, tr_predictions)))

    if config.verbose:
        plot_clusters(pipeline.steps[0][1].transform(train), tr_predictions, targets)

    return pipeline


def make_submission(pipeline):
    """
    Compose a submission using giving pipeline.
    """
    submission = pd.read_csv(config.sample_submission)

    try:
        test_df = load_data('test')
    except FileNotFoundError:
        test_df = prepare_data([os.path.join(config.test_path, 'val', path) for path in submission.wav_path])
        save_data(test_df, 'test')

    test = np.array(handle_wrong_rows(test_df, 'test'))
    te_predictions = pipeline.predict(test)

    submission['target'] = te_predictions
    submission.to_csv(config.submission, index=False)


def plot_clusters(train, predictions, targets):
    """
    Transform highly correlated with predictions features from train to 2D space and plot them for showing clusters.
    """
    features = np.array([train[:, i] for i in range(train.shape[-1])
                         if abs(np.corrcoef(predictions, train[:, i])[0, 1]) > 0.5]).T
    num = 10000
    pca = PCA(n_components=2).fit(features[:1000])
    points = pca.transform(features[:num])

    labels = [{0: 'speech', 1: 'music', 2: 'noise'}[label] for label in targets[:num]]
    palette = {'speech': 'm', 'music': 'b', 'noise': 'y'}

    plt.figure(figsize=(6, 6))
    plt.title("Clusters after dimensionality reduction")
    sns.scatterplot(points[:, 0], points[:, 1], hue=labels, palette=palette, alpha=0.15)
    plt.show()


if __name__ == '__main__':
    make_submission(train_pipeline())

