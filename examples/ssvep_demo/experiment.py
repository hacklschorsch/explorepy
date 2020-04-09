# -*- coding: utf-8 -*-
"""
A script to run a simple offline SSVEP Experiment with Mentalab's Explore device
"""

import sys
import argparse
import explorepy
from ssvep import SSVEPExperiment


def main():
    parser = argparse.ArgumentParser(description="A script to run a simple SSVEP Experiment")
    parser.add_argument("-n", "--name", dest="name", type=str, help="Name of the device.")
    parser.add_argument("-f", "--filename", dest="filename", type=str, help="Record file name")
    args = parser.parse_args()

    target_pos = [(-16, 0), (16, 0)]  # [(-14, -10), (-14, 10), (14, 10), (14, -10)]
    hints = [u'\u2190', u'\u2192'] # [u'\u2199', u'\u2196', u'\u2197', u'\u2198']
    fr_rates = [6, 8]  # [5, 6, 7, 8] -> 12hz - 10hz - 8.5hz - 7.5hz
    exp_device = explorepy.Explore()
    exp_device.connect(device_name=args.name)
    exp_device.record_data(file_name=args.filename, file_type='edf')

    def send_marker(idx):
        """Maps index to marker code and sends it to the Explore object"""
        code = idx + 8
        exp_device.set_marker(code)

    experiment = SSVEPExperiment(frame_rates=fr_rates, positions=target_pos, hints=hints, marker_callback=send_marker,
                                 trial_len=6, trials_per_block=5, n_blocks=20,
                                 screen_refresh_rate=60
                                 )
    experiment.run()
    exp_device.stop_recording()
    exp_device.disconnect()
    sys.exit(0)


if __name__ == '__main__':
    main()
