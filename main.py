import os
import config
import fnmatch
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def list_audiofiles(path):
    audiofiles = []

    for dirpath, _, files in os.walk(path):
        for file in fnmatch.filter(files, '*.wav'):
            audiofiles.append(os.path.join(dirpath, file))

    if config.verbose:
        print(f"Found {len(audiofiles)} input audiofiles: ..., {audiofiles[-1]}")

    return audiofiles


def load_data(dataset):
    loaded = pd.read_csv(os.path.join(config.prepared_path, dataset) + '.csv')
    if config.verbose:
        print(f"Loaded {dataset}: {len(loaded)} x {len(loaded.columns)}")

    return loaded


def extract_mfcc(data):
    window_width = 0.220
    window_stride = 0.050
    sr = config.sample_rate

    mfccs = []
    for n_mfcc in [11, 5, 2]:
        mfccs.append(librosa.feature.mfcc(data,
                                          sr=sr,
                                          n_mfcc=n_mfcc,
                                          dct_type=2,
                                          hop_length=int(window_stride * sr),
                                          n_fft=int(window_width * sr)))

    mfcc_means = np.concatenate([np.mean(mfccs[i], axis=1) for i in range(3)])
    mfcc_std = np.concatenate([np.std(mfccs[i], axis=1) for i in range(3)])

    return np.concatenate((mfcc_means, mfcc_std)).flatten()


def extract_brightness(data):
    frequency_threshold = 1800
    fft = np.abs(np.fft.rfft(data))
    return [np.sum(fft[frequency_threshold:]) / np.sum(fft)]


def extract_rolloff(data):
    fft = np.abs(np.fft.rfft(data))
    threshold = np.sum(fft) * 0.85
    energy = 0.0

    for i in range(fft.size):
        energy += fft[i]
        if energy >= threshold:
            return [i]


def extract_stat(data):
    return [np.percentile(row, per) for row in librosa.feature.mfcc(y=data, sr=config.sample_rate, n_mfcc=40)
            for per in (1, 25, 50, 75, 99)]


def prepare_data(audiofiles, dataset):
    data = []

    for file in audiofiles:
        wav, _ = sf.read(file)
        row = []
        for feature in config.features:
            row.extend(globals()[f'extract_{feature}'](wav))

        data.append(row)

    columns = [feature[0] + str(i) for feature in config.features
               for i in range({'m': 36, 'b': 1, 'r': 1, 's': 200}[feature[0]])]
    prepared = pd.DataFrame(data, columns=columns)
    prepared.to_csv(os.path.join(config.prepared_path, dataset) + '.csv', index=False)

    if config.verbose:
        print(f"Prepared {dataset}: {len(prepared)} x {len(prepared.columns)}")

    return prepared


def handle_wrong_rows(data, dataset):
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


def relabel(predictions, re_dict):
    re_predictions = np.zeros_like(predictions)

    for key in re_dict:
        re_predictions[predictions == key] = re_dict[key]

    return re_predictions


if __name__ == '__main__':

    try:
        train_df = load_data('train')
    except FileNotFoundError:
        train_df = prepare_data(list_audiofiles(config.train_path), 'train')

    submission = pd.read_csv(config.sample_submission)

    try:
        test_df = load_data('test')
    except FileNotFoundError:
        test_df = prepare_data([os.path.join(config.test_path, 'val', path) for path in submission.wav_path], 'test')

    train = np.array(handle_wrong_rows(train_df, 'train'))
    test = np.array(handle_wrong_rows(test_df, 'test'))

    scaler = MinMaxScaler().fit(train)
    km = KMeans(n_clusters=config.n_classes, n_init=200, max_iter=10000).fit(scaler.transform(train))
    predictions = km.predict(scaler.transform(test))

    print(f"20 first predictions are\n {predictions[:20]}")
    print("Enter values for relabeling")

    re_dict = dict()
    for c in range(config.n_classes):
        re_dict[c] = int(input(f"{c}: "))

    re_predictions = relabel(predictions, re_dict)
    submission['target'] = re_predictions
    submission.to_csv(config.submission, index=False)
