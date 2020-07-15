import mne
import numpy as np

import matplotlib.pyplot as plt

from mne.time_frequency import tfr_morlet
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.viz.utils import center_cmap

from analysis import CCAAnalysis

filename = "Masoome_07-04-2020_16-30_ExG.bdf"
ch_names = ['POz', 'Oz', 'O1', 'O2']
data = mne.io.read_raw_bdf(input_fname=filename)
data.load_data()
event_ids = {'8': 0, '9': 1}  # , '10': 2, '11': 3}
event_freq = [10, 7.5]  # [12, 10, 8.5, 7.5]
events, _ = mne.events_from_annotations(data, event_ids)
data.notch_filter(50)
data.filter(l_freq=1, h_freq=35)
tmin, tmax = -1.5, 6
picks = mne.pick_channels(data.info["ch_names"], ["ch1", "ch2", "ch3", "ch4"])
epochs = mne.Epochs(data, events=events, event_id=event_ids, picks=picks, tmin=tmin, tmax=tmax)

# compute ERDS maps ###########################################################
freqs = np.arange(2, 28, .2)
n_cycles = freqs  # use constant t/f resolution
vmin, vmax = -1, 10
tmin, tmax = -1, 6  # set min and max ERDS values in plot
baseline = [-1, 0]  # baseline interval (in s)
cmap = center_cmap(plt.cm.RdBu, vmin, vmax)  # zero maps to white

# Run TF decomposition overall epochs
tfr = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                 use_fft=True, return_itc=False, average=False,
                 decim=2)
tfr.crop(tmin, tmax)
tfr.apply_baseline(baseline, mode="percent")
for event in event_ids:
    tfr_ev = tfr[event]
    fig, axes = plt.subplots(1, 5, figsize=(12, 4),
                             gridspec_kw={"width_ratios": [10, 10, 10, 10, 1]})
    for ch, ax in enumerate(axes[:-1]):
        tfr_ev.average().plot([ch], vmin=vmin, vmax=vmax, cmap=(cmap, False),
                              axes=ax, colorbar=False, show=False)

        ax.set_title(ch_names[ch], fontsize=10)
        ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
        if not ax.is_first_col():
            ax.set_ylabel("")
            ax.set_yticklabels("")
    fig.colorbar(axes[0].images[-1], cax=axes[-1])
    fig.suptitle("ERS (Target: {} Hz)".format(event_freq[event_ids[event]]))
    fig.show()

t_win = [.25, .5, .75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 6]
tmin = 0
epochs = mne.Epochs(data, events=events, event_id=event_ids, picks=picks, tmin=tmin, tmax=tmax, baseline=(0, 0))

preds = {str(key): [] for key in t_win}
accuracies = []
for tmax in t_win:
    cca = CCAAnalysis(freqs=event_freq, win_len=tmax, s_rate=250, n_harmonics=2)
    for eeg_chunk in epochs:
        scores = cca.apply_cca(eeg_chunk[:, 0:int(tmax * 250)].T)
        preds[str(tmax)].append(np.argmax(scores))
    accuracies.append(np.count_nonzero(np.array(preds[str(tmax)]) == epochs.events[:, 2]) / len(preds[str(tmax)]) * 100)
print(accuracies)
fig, ax = plt.subplots()
ax.plot(t_win, accuracies, marker='*', markersize=5)
ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Time window (s)')
ax.set_xlim(0, 6.5)
ax.set_ylim(40, 100)
ax.grid(True)
plt.show()
print('s')
