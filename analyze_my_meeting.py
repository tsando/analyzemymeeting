import numpy as np
import pandas as pd
import librosa  # audio analysis library
from librosa import display  # for plotting mel spectogram
import handy_functions  # for sliding_window function for np.array
from importlib import reload
reload(handy_functions)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sns.set(style="whitegrid", rc={"figure.figsize": (12, 4),
                               #                                "legend.fontsize": "large",
                               #                                "axes.titlesize": "x-large",
                               #                                "xtick.labelsize": "x-large",
                               #                                "ytick.labelsize": "x-large",
                               })

###########################################################################
###########################################################################


def plot_speaker_timeline_truth_and_pred(df):
    # Speaker timeline
    df[['kmeans', 'truth']].plot()
    plt.yticks(np.unique(df['truth']).astype(int))
    plt.ylabel('Speaker number ID')
    plt.xlabel('Time (seconds)')
    plt.legend()
    plt.show()
    pass


def plot_speaker_timeline_from_pred(t, labels):
    # Use the previously generated time range for the shape of X to plot labels vs time
    plt.plot(t, labels)
    plt.yticks(np.unique(labels).astype(int))
    plt.xlim([0, t[-1]])
    plt.ylabel('Speaker number id')
    plt.xlabel('Time (seconds)')
    plt.show()
    pass


def plot_stfeatures():
    # generates the correct time labels for mfccs shape
    t = librosa.frames_to_time(
        np.arange(mfccs.shape[1]), sr=sr, hop_length=frame_step*sr)
    librosa.display.specshow(mfccs, x_axis='time', x_coords=t)
    plt.colorbar()
    plt.title('Unnormalised MFCCs (short-term features)')
    plt.tight_layout()
    pass


def plot_mtfeatures():

    # Plot NORMALISED mid-term features (rolling mean and std of mels)
    # note here we need to manually calculate the x_coords given shampe of input array and in this case no need to input sr
    sr_X = (X_scaled.shape[0] - 1)/duration
    t = librosa.samples_to_time(np.arange(X_scaled.shape[0]), sr=sr_X)
    librosa.display.specshow(X_scaled.T, x_axis='time', x_coords=t)
    plt.colorbar()
    plt.title('Normalised mid-term features (mean)')
    plt.tight_layout()


def make_all_plots(y, sr):
    # Plot the sound wave
    librosa.display.waveplot(y, sr=sr)
    # Mel spectogram
    plot_stfeatures()
    plot_mtfeatures()
    pass


def get_truth_df():
    # Compare to ground-truth to see the overall performance of the classification
    truth = pd.read_csv(
        'data/diarization/diarizationExample.segments', names=['start', 'end', 'speaker'])
    # First map speakers to same labels from k-means
    truth['labels'] = np.where(truth['speaker'] == 'speakerA', 2,
                               np.where(truth['speaker'] == 'speakerB', 3,
                                        np.where(truth['speaker'] == 'speakerC', 1,
                                                 np.where(truth['speaker'] == 'speakerD', 0, np.nan)))).astype(int)
    # Upsample labels to match time array shape
    d = truth[['start', 'labels']].to_dict(orient='list')
    # This is an O(n) solution
    speakers = []
    i = 0
    for it in t:
        if i + 1 >= len(d['start']):
            speakers += [d['labels'][i]]
        elif it < d['start'][i + 1]:
            speakers += [d['labels'][i]]
        else:
            i += 1
            speakers += [d['labels'][i]]
        # print(i, it, len(speakers))  # for debugging
    df = pd.DataFrame(
        index=t, data={'truth': speakers, 'kmeans': labels.tolist()})
    return df


def get_meeting_stats(df):
    speaking_time = calc_speaking_time(df)
    # plot_speaking_time(speaking_time)
    n_times_spoken = calc_n_times_spoken(df)
    return {'speaking_time': speaking_time, 'n_times_spoken': n_times_spoken}


def calc_speaking_time(df):
    df['time'] = df.index
    t_step = df['time'].diff(1).iloc[-1]
    speaking_time = (df.groupby('kmeans').count()[
                     ['time']] * t_step).sort_values(by=['time'], ascending=False)
    return speaking_time


def plot_speaking_time(speaking_time):
    # Stacked bar chart
    speaking_time.T.plot.bar(stacked=True)
    # Pie chart of percentage time
    speaking_time['time'].plot.pie(labels=['speaker ' + str(x) for x in speaking_time.index.tolist()],
                                   autopct='%.f',
                                   figsize=(6, 6))


def calc_n_times_spoken(df):
    return df[df['kmeans'].diff(1) != 0].groupby('kmeans').count().sort_values(by='time', ascending=False)


###########################################################################
###########################################################################

# Inputs
fn = 'data/diarization/diarizationExample.wav'  # .wav file location
sr = 16000  # sampling freq
# n_fft in librosa
frame_size = 50e-3  # 50 ms
# hop_length in librosa
frame_step = 25e-3  # 25 ms
# n speakers
nspeakers = 4

# Load with librosa with 16 kHz sampling freq (same as in pyAudioAnalysis)
y, sr = librosa.load(fn, sr=sr)
duration = librosa.get_duration(y, sr=sr)

# Normalisation of signal using same methodology as pyAudioAnalysis
y = (y - y.mean()) / ((np.abs(y)).max() + 0.0000000001)

# Short-term features

# Calculate short-term mfcss with the above frame size and step
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=int(
    frame_size*sr), hop_length=int(frame_step*sr))

# Mid-term features

# Get the array split into sliding windows of 80 samples each and step=8 as in pyAudioAnalysis
mfccs_strided = handy_functions.sliding_window(mfccs, size=80, stepsize=8)
# Get rolling mean (note using axis=2)
_mean = np.mean(mfccs_strided, axis=2)
# Get rolling std (note using axis=2)
_std = np.std(mfccs_strided, axis=2)
# Join mean and std into single array
X = np.concatenate((_mean, _std))
# Get transpose to put into (n_samples, n_features) shape
X = X.T

# Normalise feature matrix

# Scale to zero mean and unit std
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Infer the sampling rate by dividing the shape minus 1 divided by duration (only this matches the duration)
sr_X = (X_scaled.shape[0] - 1)/duration
t = librosa.samples_to_time(np.arange(X_scaled.shape[0]), sr=sr_X)

# Unsupervised Classification

# Assuming 4 speakers (which is also the ground truth)
kmeans = KMeans(n_clusters=nspeakers, random_state=0)
# Note that X is in form (n_samples, n_features)
labels = kmeans.fit_predict(X_scaled)

# plot_speaker_timeline_from_pred(t, labels)

df = get_truth_df()

print(get_meeting_stats(df))
