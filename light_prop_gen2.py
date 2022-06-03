import numpy as np
from optipropapy import optipropapy_functions as opp
import matplotlib.pyplot as plt
from threading import Thread
from datetime import date
import time
import uuid
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


def eeng716_final(save_flag, plot_flag, zern, ch, nn, d, ro):
    # save file variables
    start_time = time.time()
    today = date.today()
    root = './data/'
    file_prefix = ''+today.strftime('%Y-%m-%d') + \
               f'_{np.random.randint(0, 9)}{np.random.randint(0, 9)}{np.random.randint(0, 9)}_'
    print('    '+file_prefix)

    # scenario variables
    l1 = d * 2  # edge length of image (2 * telescope diameter)
    frames = 32  # image frames
    z = 10  # focal length distance
    lam = 0.5  # wavelength of light

    # generate needed data
    sources = 3
    lu1 = 2000
    lu2 = 1000
    locs = np.array([[0, 5], [5, 0], [3, 4], [4, 3]])
    loc = locs[np.random.randint(0, 4), :]
    xoff =loc[0]
    yoff = loc[1]
    deltat = .002

    source = np.zeros((sources, frames, nn, nn))
    source[0, :, :, :] = np.zeros((frames, nn, nn)) * 20
    source[1, :, :, :] = make_source(0, 0, frames, nn, 2000, 1000, 20)
    source[2, :, :, :] = make_source(xoff, yoff, frames, nn, 2000, 1000, 20)
    scene_info = np.array([[0, 1, lu2/lu1],[1, 1, lu2/lu1],[2, 1, lu2/lu1]])
    phase_screens, x_crd, zern = make_screens(frames, nn, d, ro, l1, deltat, zern, ch, zern_mx)
    z4 = zern[4-2, :, :]  # minus 2 to account for 1) zero indexing and 2) removing piston (z1)
    screens = np.exp(1j * phase_screens)
    pupil = opp.circ_mask(nn, l1, d, 0)

    # instantiate propagation variables
    otfs = np.zeros((nn, nn, frames)) + 0j
    psfs = np.zeros((nn, nn, frames))
    s_scrnd = np.zeros((sources, frames, nn, nn))
    s_phtns = np.zeros((sources, frames, nn, nn))
    r = np.zeros((sources, frames, nn, nn)) + 0j

    # calculate psfs and otfs
    for ff in range(frames):
        # generate otf and psf from phase screen and pupil function
        screen = np.fft.fftshift(screens[:, :, ff] * pupil)
        psfs[:, :, ff], otfs[:, :, ff] = opp.convert_to_psf_otf(screen)


    # setup a function to run in a multi-threaded process (as each max-likelihood iteration takes awhile)
    def data_thread(ss):
        # make data
        for ff in range(frames):
            # propagate from optic to receiver after multiplying the otf
            source[ss, ff, :, :] = np.fft.fftshift(source[ss, ff, :, :])
            r[ss, ff, :, :], xr = opp.distort_and_focus(source[ss, ff, :, :], otfs[:, :, ff], z, lam, x_crd)

            # get magnitude of receive field
            s_scrnd[ss, ff, :, :] = np.abs(r[ss, ff, :, :])

            # add random variables (biased by 40 to remove <0 values due to added normal rv)
            s_phtns[ss, ff, :, :] = np.random.poisson(s_scrnd[ss, ff, :, :]) + \
                                    np.random.normal(0, 1, size=(nn, nn))
            s_phtns[ss, ff, :, :] = np.fft.fftshift(s_phtns[ss, ff, :, :])
        return True

    # create a thread/process per different source and run/start
    threads = []
    for pp in range(sources):
        process = Thread(target=data_thread, args=[pp])
        process.start()
        threads.append(process)

    # join allows all threads to complete/terminate before moving on
    for process in threads:
        process.join()

    # initialize phase unwrap variables
    zern_true = np.zeros((np.shape(zern)[0], frames))

    # get true zernike coefficients
    for ff in range(frames):
        for zz in range(np.shape(zern)[0]):
            zern_true[zz, ff] = np.sum(np.sum(phase_screens[:, :, ff] * zern[zz, :, :])) / \
                                np.sum(np.sum(zern[zz, :, :] ** 2))

    # plot final propagation when true, can access as gif in folder
    if plot_flag:
        animate_results(source, screens, s_phtns, frames)

    # output time for simulation instance
    print('    time: ', (time.time() - start_time))
    labels = scene_info
    path = './gen_data'
    if save_flag:
        save_file(path, s_phtns, labels)

    print('    done')


# save data to file for later use
def save_file(path, data, labels):
    # convert images to bytes then saves (frames, 140px, 146px)
    for ii in range(3):
        data_min = np.min(np.min(data[ii]))
        data_max = np.max(np.max(data[ii]))
        data[ii] = (data[ii] - data_min) / (data_max - data_min) * 255
        d = data[ii].astype('uint8')
        np.save(file=path+'/'+str(int(labels[ii, 0]))+'/'+str(uuid.uuid4()), arr=d)

    print('    saved data')


# Create phase screens
def make_screens(frames, nn, d, ro, l1, deltat, zern, ch, zern_mx):
    # define variables
    x_crd = np.linspace(-l1 / 2, l1 / 2, nn)  # create coordinates
    windx = 6
    windy = 6
    boil = 1

    # generate screens
    screen = opp.zern_phase_scrn(ro, nn, zern_mx, x_crd, windx, windy, boil, deltat, frames, zern, ch, True)

    z = np.zeros((zern_mx-1, nn, nn))
    z[:, int(nn / 4):int(3 * nn / 4), int(nn / 4):int(3 * nn / 4)] = zern
    return screen, x_crd, z


# Create source images
def make_source(x_off, y_off, frames, nn, lu, lu2, back):
    # define variables
    lu = lu - back
    lu2 = lu2 - back

    # generate object
    if (x_off == 0) and (y_off == 0):
        source = opp.gen_sourcefield_pts(nn, 0, 0, lu)
    else:
        obj1 = opp.gen_sourcefield_pts(nn, 0, 0, lu)
        obj2 = opp.gen_sourcefield_pts(nn, x_off, x_off, lu2)
        source = obj1 + obj2 + back

    # return the created object repeated <frame> number of times
    return np.repeat(source[np.newaxis, :, :], frames, axis=0)


# create animated gif of the data to show the effects of atmosphere over time
def animate_results(field_s1, screens, data, frames):
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(5, 5, True)
    for ii in range(3):
        data_min = np.min(np.min(data[ii, :, :, :]))
        data_max = np.max(np.max(data[ii, :, :, :]))
        data[ii, :, :, :] = (data[ii, :, :, :] - data_min) / (data_max - data_min) * 255
        data[ii, :, :, :] = data[ii, :, :, :].astype(dtype='uint8')

    def AnimationFunction(f):
        ax[0, 0].imshow(field_s1[2, f, :, :])
        ax[0, 0].set_title('source (2)')
        ax[0, 0].axis('off')
        ax[1, 0].imshow(data[1, f, :, :])
        ax[1, 0].axis('off')
        ax[0, 1].imshow(np.real(screens[:, :, f]))
        ax[0, 1].set_title('atmos')
        ax[0, 1].axis('off')
        ax[1, 1].imshow(data[2, f, :, :])
        ax[1, 1].axis('off')

    fps = 10
    ani = FuncAnimation(fig, AnimationFunction, frames=frames, interval=50)
    writer = PillowWriter(fps=fps)
    ani.save('light_gen.gif', writer=writer)


if __name__ == '__main__':
    it = 9000  # number of iterations of simulation
    zern_mx = 55  # number of zernikes
    nn = 100  # samples along a single frame edge (add 1 to make center pxl)
    d = 0.5  # diameter of pupil function
    ro = 0.05  # frieds seeing parameter
    zern, ch = opp.generate_zern_polys(zern_mx, int(nn / 2), d, ro)
    for ii in range(it):
        eeng716_final(True, False, zern, ch, nn, d, ro)
        print(f'It: {ii}/{it}')
