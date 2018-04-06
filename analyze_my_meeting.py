import argparse
import os
import numpy as np
import pandas as pd
import librosa  # audio analysis library
import handy_functions  # for sliding_window function for np.array
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


###########################################################################
###########################################################################


def save_meeting_stats(stats, outdir):
    stats['timeline'].index.name = 'time'
    for k in stats:
        stats[k].to_csv('{}/{}.csv'.format(outdir, k))


def get_meeting_stats(df, labelAccessor):
    stats = {'timeline': df,
             'speaking_time': calc_speaking_time(df, labelAccessor),
             'n_times_spoken': calc_n_times_spoken(df, labelAccessor)}
    print(stats)
    return stats


def calc_speaking_time(df, labelAccessor):
    df0 = df.copy()
    df0['speaking_time'] = df0.index
    t_step = df0['speaking_time'].diff(1).iloc[-1]
    df0 = (df0.groupby(labelAccessor).count()[
               ['speaking_time']] * t_step).sort_values(by=['speaking_time'], ascending=False)
    return df0


def calc_n_times_spoken(df, labelAccessor):
    df0 = df.copy()
    df0['n_times_spoken'] = 1
    df0 = df0[df0[labelAccessor].diff(1) != 0].groupby(
        labelAccessor).count().sort_values(by='n_times_spoken', ascending=False)
    return df0


###########################################################################
###########################################################################

def analyzemymeeting(indir, nspeakers, sr=16000,
                       frame_size=50e-3, frame_step=25e-3, outdir=False):
    """
    Parameters
    ----------
    indir  :   .wav file location e.g. 'data/diarization/diarizationExample.wav'
    sr  :   sampling freq, default=16kHz
    frame_size  :   n_fft in librosa, default 50 ms
    frame_step  :   hop_length in librosa, default 25 ms
    nspeakers   :   number of speakers in meeting (int)

    """
    # Identify .wave file from input dir (WARNING: assuming only one file in the directory!!)
    fn = os.path.join(indir, os.listdir(indir)[0])

    # Load with librosa with 16 kHz sampling freq (same as in pyAudioAnalysis)
    y, sr = librosa.load(fn, sr=sr)
    duration = librosa.get_duration(y, sr=sr)

    # Normalisation of signal using same methodology as pyAudioAnalysis
    y = (y - y.mean()) / ((np.abs(y)).max() + 0.0000000001)

    # Short-term features

    # Calculate short-term mfcss with the above frame size and step
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=int(
        frame_size * sr), hop_length=int(frame_step * sr))

    # Mid-term features

    # Get the array split into sliding windows of 80 samples each and step=8
    # as in pyAudioAnalysis
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

    # Infer the sampling rate of the final shape by dividing the shape minus 1
    # divided by duration (only this matches the duration)
    sr_X = (X_scaled.shape[0] - 1) / duration
    t = librosa.samples_to_time(np.arange(X_scaled.shape[0]), sr=sr_X)

    # Unsupervised Classification

    # Assuming 4 speakers (which is also the ground truth)
    kmeans = KMeans(n_clusters=nspeakers, random_state=0)
    # Note that X is in form (n_samples, n_features)
    labels = kmeans.fit_predict(X_scaled)

    # Create df of results to get stats and visualizations
    df = pd.DataFrame(index=t, data={'speaker_number_id': labels})

    stats = get_meeting_stats(df, 'speaker_number_id')
    if outdir:
        save_meeting_stats(stats, outdir)

    # Remove file from uploads directory
    os.remove(fn)
    # df = get_truth_df()
    pass


def main():
    """
    Example inputs:
    indir  :   '/Users/tsando/code/express-app/app-ejs/public/data/uploads'
    nspeakers   :   4

    Usage:  python analyzemymeeting.py /Users/tsando/code/express-app/app-ejs/public/data/uploads 4 /Users/tsando/code/express-app/app-ejs/public/data/csvs
    """

    # analyzemymeeting(indir='/Users/tsando/code/express-app/app-ejs/public/data/uploads',
    #                    nspeakers=4,
    #                    outdir='/Users/tsando/code/express-app/app-ejs/public/data/csvs')

    parser = argparse.ArgumentParser()
    parser.add_argument("indir", help="Input directory containing .wav file")
    parser.add_argument("nspeakers", help="Number of speakers in meeting", type=int)
    parser.add_argument("outdir", help="Output directory")
    args = parser.parse_args()
    analyzemymeeting(indir=args.indir, nspeakers=args.nspeakers, outdir=args.outdir)

    pass


if __name__ == "__main__":
    main()
    pass
