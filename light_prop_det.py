import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from optipropapy import optipropapy_functions as opp
from threading import Thread
from datetime import date


def light_prop_det():
    instance = '2022-06-03_864'

    # variables to access data
    base_data_dir = Path('gen_data')
    load_dir = 'data_ests'

    # load processed data
    obj_fn = instance + '_objEst.npy'
    truth_fn = instance + '_truth.npy'
    otf_fn = instance + '_otfEst.npy'
    obj_est = np.load(base_data_dir / load_dir / obj_fn)
    truth = np.load(base_data_dir / load_dir / truth_fn)
    truth = truth.astype('float')
    otf_est = np.load(base_data_dir / load_dir / otf_fn)

    # additional derived variables
    runs = np.shape(obj_est)[0]
    print(f'Runs: {runs}, Truth: {truth.dtype}')

    # check to make sure referencing data correctly
    if True:
        for rr in range(runs):
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(np.fft.fftshift(obj_est[rr, :, :]))
            ax[1].imshow(np.fft.fftshift(np.abs(otf_est[rr, :, :, 0])))
            fig.suptitle(f'Truth: {truth[rr]}')
            plt.show()

    # detect
    divs = 100
    thresh = np.linspace(0, 1, divs)
    det = np.ones((runs, divs))
    dfa = np.zeros((2, divs))
    p2_ratio = np.zeros(runs)
    x = np.linspace(0, 99, 100)
    xm, ym = np.meshgrid(x, x)
    for rr in range(runs):
        # get center of object
        cx = np.sum(obj_est[rr, :, :] * xm) / np.sum(obj_est[rr, :, :])
        cy = np.sum(obj_est[rr, :, :] * ym) / np.sum(obj_est[rr, :, :])

        # create detection areas
        outer = opp.circ_mask_shift(100, 100, 50, 3, -(cx - 50), -(cy - 50))
        inner = opp.circ_mask_shift(100, 100, 3, 0, -(cx - 50), -(cy - 50))

        p2 = np.sum(outer * np.fft.fftshift(obj_est[0, :, :]))
        p1 = np.sum(inner * np.fft.fftshift(obj_est[0, :, :]))
        p2_ratio[rr] = p2 / (p1 + p2)

    for tt in range(divs):
        detection = 0
        false_alarm = 0
        for rr in range(runs):
            if p2_ratio[rr] > thresh[tt]:
                det[rr, tt] = 2
                if truth[rr] == 2:
                    detection = detection + 1
                if truth[rr] == 1:
                    false_alarm = false_alarm + 1
        print(f'FA: {false_alarm/runs}, D: {detection/runs}')
        dfa[0, tt] = false_alarm / runs
        dfa[1, tt] = detection / runs
        print(f'Div: {tt}/{divs}')

    # plot ROC and Pfa & Pd distributions
    if True:
        fig, ax = plt.subplots(1, 3)
        ax[0].plot(dfa[0, tt], dfa[1, tt])
        ax[1].plot(dfa[0, tt])
        ax[1].plot(dfa[1, tt])
        ax[2].plot(p2_ratio)
        plt.show()

    print(f'P1: {p1}, P2: {p2}')

    # view detection masking areas
    if False:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(outer)
        ax[1].imshow(inner)
        plt.show()

    print('done')


if __name__ == '__main__':
    light_prop_det()
