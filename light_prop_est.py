import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from optipropapy import optipropapy_functions as opp
from threading import Thread
from datetime import date


def light_prop_est():
    # variables to access data
    base_data_dir = Path('/opt', 'data', 'gen_data')
    data_dir = base_data_dir / 'raw'
    sub_dir = 'train'
    save_path = './gen_data/data_ests'

    # get all sub-directories
    dirs = os.listdir(data_dir / sub_dir)
    n = 0
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

    # get some needed variables
    nn = 100
    zern_mx = 4
    its = 200
    gs_its = 20
    frames = 32
    zern, ch = opp.generate_zern_polys(zern_mx, int(nn/2), 0.5, 0.05)
    z4 = np.zeros((nn, nn))
    z4[int(nn / 4):int(3 * nn / 4), int(nn / 4):int(3 * nn / 4)] = zern[4-2, :, :]

    # deconvolution thread
    processes = 12
    proc_groups = 2
    n = processes * proc_groups  # only use subsection of data
    obj_est = np.zeros((n, nn, nn))
    img_est = np.zeros((n, nn, nn, frames))
    otf_est = np.zeros((n, nn, nn, frames)) + 0j
    def deconv_thread(proc, data, num):
        obj_est[num, :, :], img_est[num, :, :, :], otf_est[num, :, :, :] = estimate(data, z4, its, gs_its, f'Im {num}')
        print(f'Data: {num + 1}/{n}, Thread: {proc + 1}/{processes}')

    # create threads
    for p in range(proc_groups):
        # create a thread/process per different source and run/start
        threads = []
        for pp in range(processes):
            data = np.load(files[(p * processes + pp)])
            process = Thread(target=deconv_thread, args=[pp, data, (p * processes + pp)])
            process.start()
            threads.append(process)

        # join allows all threads to complete/terminate before moving on
        for process in threads:
            process.join()

    # generate random id and save off results under this id
    today = date.today()
    id = today.strftime('%Y-%m-%d') +f'_{np.random.randint(0,9)}{np.random.randint(0,9)}{np.random.randint(0,9)}'
    np.save(file=save_path + '/' + id + '_objEst.npy', arr=obj_est)
    np.save(file=save_path + '/' + id + '_otfEst.npy', arr=otf_est)
    np.save(file=save_path + '/' + id + '_truth.npy', arr=truth[:n])
    print('done')


def estimate(data, defocus, its, gs_its, name):
    nn = 100
    frames = 32
    obj_est, img_est, otf_est = \
        opp.maxlikelihood_deconvolution2(nn, frames, data, defocus, its, gs_its, name)
    return obj_est, img_est, otf_est


if __name__ == '__main__':
    light_prop_est()
