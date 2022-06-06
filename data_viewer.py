import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from optipropapy import optipropapy_functions as opp
from threading import Thread
from datetime import date


def main():
    est_viewer()
    #gen_viewer()
    print('done')

def gen_viewer():
    samples = 4

    # variables to access data
    base_data_dir = Path('/opt', 'data', 'gen_data')
    data_dir = base_data_dir / 'raw'
    sub_dir = 'train'

    # get all sub-directories and randomize in the same way as the est script
    dirs = os.listdir(data_dir / sub_dir)
    n = 0
    sz = 100
    frames = 32
    files = []
    truth = []
    for dd in dirs:
        if '0' not in dd:
            n = n + np.size(os.listdir(data_dir / sub_dir / dd))
            for ff in os.listdir(data_dir / sub_dir / dd):
                files.append(data_dir / sub_dir / dd / ff)
                truth.append(dd)
    temp = list(zip(files, truth))
    np.random.shuffle(temp)
    files, truth = zip(*temp)

    data = np.zeros((samples, frames, sz, sz))

    for ii in range(samples):
        data[ii, :, :, :] = np.load(files[ii])
        fig, ax = plt.subplots(1, 1)
        ax.imshow(data[ii, 0, :, :])
        plt.suptitle(f'Truth: {truth[ii]}')
        plt.show()

    print('done')


def est_viewer():
    instance = '2022-06-06_225'
    samples = 12

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

    # show data from 1 -> # samples
    if np.size(truth) < samples: samples = np.size(truth)
    for ii in range(samples):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.fft.fftshift(obj_est[ii, :, :]))
        ax[1].imshow(np.fft.fftshift(np.abs(otf_est[ii, :, :, 0])))
        plt.suptitle(f'Truth: {truth[ii]}')
        plt.show()


def old_functions():
    name_data = 'light_prop_data_array.npy'
    name_label = 'light_prop_label_array.npy'

    if os.path.isfile(name_data):
        n = 50
        data_handle = np.load(name_data, allow_pickle=True)
        label_handle = np.load(name_label, allow_pickle=True)

        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(data_handle[n * 4])
        ax[0, 0].set_title('Objs: %i' % label_handle[n * 4])
        ax[0, 0].axis('off')

        ax[0, 1].imshow(data_handle[n * 4 + 1])
        ax[0, 1].set_title('Objs: %i' % label_handle[n * 4 + 1])
        ax[0, 1].axis('off')

        ax[1, 0].imshow(data_handle[n * 4 + 2])
        ax[1, 0].set_title('Objs: %i' % label_handle[n * 4 + 2])
        ax[1, 0].axis('off')

        ax[1, 1].imshow(data_handle[n * 4 + 3])
        ax[1, 1].set_title('Objs: %i' % label_handle[n * 4 + 3])
        ax[1, 1].axis('off')

        fig.suptitle('Samples = %i, n = %i:%i' % (np.size(data_handle), n * 4, n * 4 + 3))

        plt.tight_layout()
        plt.show()
    else:
        print('no file exists')


if __name__ == '__main__':
    main()
