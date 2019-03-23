import numpy
import scipy
from scipy.signal import butter, lfilter


class Filter:
    def __init__(self):
        self.cutoffA = 0.5
        self.cutoffB = 40
        self.sample_frequency = 250.0
        self.order = 5
        self.cutOffLow = 80

    def set_bandpass(self, a, b, fs, order):
        self.cutoffA = a
        self.cutoffB = b
        self.sample_frequency = fs
        self.order = order

    def butter_bandpass(self):
        nyq = 0.5*self.sample_frequency
        low_a = self.cutoffA/nyq
        high_b = self.cutoffB/nyq
        b, a = butter(self.order, [low_a, high_b], btype='band')
        return b, a

    def apply_band(self, data):
        b, a = self.butter_bandpass()
        filt_data = lfilter(b, a, data)
        return filt_data

    def set_lowpass(self, a, fs, order):
        self.cutOffLow = a
        self.sample_frequency = fs
        self.order = order

    def butter_lowpass(self):
        nyq = 0.5 * self.sample_frequency
        low = self.cutoffA / nyq
        b, a = butter(self.order, low, btype='low')
        return b, a

    def apply_lowpass(self, data):
        b, a = self.butter_lowpass()
        filt_data = lfilter(b, a, data)
        return filt_data

    def is_not_used(self):
        pass