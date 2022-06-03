import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.interpolate as itrp
import scipy.linalg as la
from scipy import signal
import csv
import time
import os
import uuid

from matplotlib.animation import FuncAnimation, PillowWriter


def light_prop_gen(nn, d, zern_mx, ro, zern, ch):
    # code flags
    save_flag = True
    plot_flag = False
    barrel_prop_flag = False

    # vars constant between simulations
    n_r1 = nn  # number of samples along 1D receive plane
    outer_d = d  # outer diameter of telescope (m)

    # start clock for running simulation instance
    start_time = time.time()

    # constants
    c = 2.9979e8  # celeritas

    # scene information
    lam = 529e-9  # optimal wavelength of camera (m)
    zo = 35786e3  # geosynch (m)
    x_s1 = 6e3  # edge length of source field (m)
    n_s1 = 30  # number of samples along 1D source plane
    x_r1 = 0.3  # edge length of receive field (m)
    del_t = .005  # exposure time
    intensity_max = 1e-5  # max target intensity (W/m^2)
    lum_max = intensity_max * del_t  # V/m^2 or Ws/m^2
    back_lum_mean = lum_max/16  # mean background luminosity

    # optic information
    fl = 2.8  # focal length (m)
    lens_d = 0.280  # lense diameter (m)
    inner_d = 0.095  # inner diameter of mask due to subreflector (m)

    # phase screen
    windx = 1  # wind velocity x direction
    windy = 1  # wind velocit y direction
    boil = 1  # atmospheric boil factor
    scrn_scale_fac = 0.003  # scaling factor on atmospheric impact

    # focal plane array (sensor) parameters
    n_sensx = 2448  # number of fpa pixels, x
    n_sensy = 2050  # number of fpa pixels, y
    sens_w = 9.93e-3  # fpa width
    sens_h = 8.7e-3  # fpa height
    pix_sz = 3.45e-6  # pixel size of photodetector

    # detector noise information (thermal and photon counting) to the system
    eff = 0.53  # quantum efficiency
    temp = 300  # temperature (K)
    cap = 1e-12  # circuit capacitance (F)

    # experiment variables
    frames = 32

    # create source field
    field_s1, xc_s1, scene_info = source_field(n_s1, x_s1, zo, lum_max)

    # propagate source field to front of telescope
    field_r1, xr1_crd, phsc_r1 = rayleigh_sommerfeld_prop(field_s1, lam, zo, xc_s1, n_r1, x_r1, 0, 0)

    # add atmospheric phase noise to the recieve plane
    screens = zern_phase_scrn(ro, outer_d, n_r1, zern_mx, xr1_crd, windx, windy, boil, del_t, frames, zern, ch)

    field_r1_atm = np.zeros((frames, nn+1, nn+1)) + 0j
    for ii in range(frames):
        field_r1_atm[ii, :, :] = field_r1 * np.exp(-1j * 2 * np.pi * (lam * c) * scrn_scale_fac * screens[:, :, ii])

    # add a mask depicting the telescope front-end
    mask_n = n_r1
    mask_sz = x_r1
    mask1 = circ_mask(mask_n, mask_sz, outer_d, inner_d)

    field_s2 = np.zeros((frames, nn + 1, nn + 1)) + 0j
    for ii in range(frames):
        field_s2[ii, :, :] = field_r1_atm[ii, :, :] * mask1

    # skip over barrel propagation if too slow (does not add much fidelity, and slows sim by a lot)
    if barrel_prop_flag:
        # propagate field from front of the telescope to the lense
        z1 = 610e-3  # propagation distance through telescope
        xms2, yms2 = np.meshgrid(xr1_crd, xr1_crd)

        field_r3 = np.zeros((frames, nn + 1, nn + 1)) + 0j
        for ii in range(frames):
            field_r3[ii, :, :], xr3_crd, d_r3 = fresnel_prop2(field_s2[ii, :, :], xms2, yms2, x_r1, z1, lam, n_r1)
    else:
        # do not add extra step of propagating through telescope barrel
        xr3_crd = xr1_crd
        d_r3 = x_r1 / 2 / n_r1
        field_r3 = field_s2

    # focus field based on effective lense focal length, turns fresnel into fraunhofer of distance fl
    mask2 = circ_mask(mask_n, mask_sz, lens_d, 0)

    field_s3 = np.zeros((frames, nn + 1, nn + 1)) + 0j
    for ii in range(frames):
        field_s3[ii, :, :] = field_r3[ii, :, :] * mask2

    field_r4 = np.zeros((frames, nn + 1, nn + 1)) + 0j
    for ii in range(frames):
        field_r4[ii, :, :], xr4_crd = fraunhofer_prop(field_s3[ii, :, :], lam, fl, xr3_crd, d_r3)

    # sample region on the ccd array
    sens_crdx = np.linspace(-sens_w / 2, sens_w / 2, n_sensx)
    sens_crdy = np.linspace(-sens_h / 2, sens_h / 2, n_sensy)

    data_sampled = np.zeros((frames, n_sensy, n_sensx))
    for ii in range(frames):
        data_sampled[ii, :, :] = sample(field_r4[ii, :, :], xr4_crd, sens_crdx, sens_crdy)

    # add noise based on the detector
    data_noisey = np.zeros((frames, n_sensy, n_sensx))
    for ii in range(frames):
        data_noisey[ii, :, :] = \
            detection_noise(data_sampled[ii, :, :], n_sensx, n_sensy, eff, lam, del_t, back_lum_mean, pix_sz, temp, cap)

    # output time for simulation instance
    print('time: ', (time.time() - start_time))

    # save off data when true
    data = partition_image(data_noisey, n_sensx, n_sensy, np.max(xr4_crd) / sens_w, np.max(xr4_crd) / sens_h, frames)

    labels = scene_info
    path = './gen_data'
    if save_flag:
        save_file(path, data, labels)

    # plot final propagation when true, can access as gif in folder
    if plot_flag:
        animate_results(field_s1, screens, data, labels, frames)


    print('done')


# space objects desired for detection
def source_field(n, sz, z, lu):
    x = np.arange(-sz / 2, sz / 2 + (sz / n), sz / (n))
    # xm, ym = np.meshgrid(x, x)
    field_s = np.zeros((n + 1, n + 1))

    lumin = np.random.uniform(lu/10, lu, size=(2, 2, 2))
    num_obj = np.random.randint(0, 3, size=(2, 2))

    for ii in range(2):
        for jj in range(2):
            if num_obj[ii, jj] == 0:
                # do nothing
                continue
            elif num_obj[ii, jj] == 1:
                # create one object
                field_s[int((3**ii) * n / 4), int((3**jj) * n/4)] = lumin[ii, jj, 0]
            elif num_obj[ii, jj] == 2:
                # create two objects
                field_s[int((3**ii) * n / 4), int((3**jj) * n / 4)] = lumin[ii, jj, 0]
                field_s[int(((3**ii) * n) / 4), int(((3**jj) * n) / 4) + 1] = lumin[ii, jj, 1]

    scene_info = np.concatenate((num_obj.reshape(4, 1), lumin.reshape(4, 2)), 1)
    print('source field gend')
    return field_s, x, scene_info


# circle center circle masking, simulating a subreflector
def circ_mask(n, sz, outer_d, inner_d):
    x = np.arange(-sz / 2, sz / 2 + (sz / n), sz / n)
    xm, ym = np.meshgrid(x, x)
    mask = np.zeros((n + 1, n + 1))

    mask[(xm ** 2 + ym ** 2) <= (outer_d / 2) ** 2] = 1
    mask[(xm ** 2 + ym ** 2) < (inner_d / 2) ** 2] = 0

    return mask


# account for the points measured at focal plane array
def sq_mask(n, sz, x_diam, y_diam):
    x = np.arange(-sz / 2, sz / 2 + (sz * 0.5 / n), sz / n)
    xm, ym = np.meshgrid(x, x)
    mask = np.zeros((n + 1, n + 1))

    maskx = np.abs(xm) <= x_diam / 2
    masky = np.abs(ym) <= y_diam / 2
    maski = maskx * masky
    mask[maski] = 1

    return mask


# resample to focal plane array
def sample(field_s, xs_crd, sampx, sampy):
    # determine magnitude of source field
    field_s_mag = np.sqrt(field_s.real**2 + field_s.imag**2)
    # field_s_mag = np.abs(field_s @ field_s.conj().T)

    field_rs = itrp.interp2d(xs_crd, xs_crd, field_s_mag, kind='linear')
    # field_rs2 = itrp.RectBivariateSpline(xs_crd, xs_crd, field_s_mag)

    print('resampled')
    return field_rs(sampx, sampy)


# adds noise that is typical on a light detector
def detection_noise(data, n_x, n_y, eff, lam, del_t, back_mean, px_sz, temp, cap):
    c = 2.9979e8  # celeritas
    h = 6.626e-34  # planck's constant
    qe = 1.6022e-19  # elementary charge
    kb = 1.38065e-34  # boltzmann's constant

    y_hi = int(n_x * (9.0 / 16.0))
    y_lo = int(n_x * (7.0 / 16.0))
    x_hi = int(n_y * (7.0 / 12.0))
    x_lo = int(n_y * (5.0 / 12.0))

    # add detector noise from circuit and scene
    thermal = np.sqrt(kb * temp * cap / qe ** 2)  # thermal noise
    data_signal = (data * (px_sz ** 2) * del_t) / (h * c / lam)  # field (I/m^2) to photons
    background = back_mean * del_t / (h * c / lam)  # background noise

    # only make portion of interest random to speed up simulation
    thermal_rv = np.random.normal(0, thermal, size=[int(n_y/6), int(n_x/8)])
    data_signal_rv = data_signal
    data_signal_rv[x_lo:x_hi,y_lo:y_hi] = np.random.poisson(eff * data_signal[x_lo:x_hi,y_lo:y_hi])
    background_rv = np.random.poisson(eff * background, size=[int(n_y/6), int(n_x/8)]) * 1e5

    print('added noise')
    data_signal_rv[x_lo:x_hi,y_lo:y_hi] = data_signal_rv[x_lo:x_hi,y_lo:y_hi] + thermal_rv + background_rv
    return data_signal_rv


# create phase screen from zernike polynomials
def zern_phase_scrn(ro, d, nn, zern_mx, x_crd, windx, windy, boil, deltat, frames, zern, ch):
    # generate phase screen directly from zernike poly
    #screens = general_phase_screen(nn, zern_mx, zern, ch)

    # generate phase screen using atmospheric vars
    screens = atmos_phase_screen(nn, ro, x_crd, windx, windy, boil, deltat, zern_mx, zern, ch, frames)

    print('zern scrn')
    return screens


# generates phase screens randomly from zernike polynomials
def general_phase_screen(nn, zern_mx, zern, ch):
    rn = np.random.normal(size=(zern_mx - 1, 1))
    z_cf = np.matmul(ch, rn)
    zern_phs = np.zeros((nn + 1, nn + 1))

    for ii in np.arange(0, zern_mx - 1):
        zern_phs = zern_phs + z_cf[ii] * zern[ii, :, :]

    return zern_phs


# creates phase screens based on zernike polynom,ials and atmospheric variables
def atmos_phase_screen(nn, ro, x_crd, windx, windy, boil, deltat, zern_mx, zern, ch, frames):
    a = 6.88

    # get phase structure
    xm, ym = np.meshgrid(x_crd, x_crd)  # tau_x, tau_y
    phs_struct = a * (((ym + (windy + boil) * deltat) ** 2 + (xm + (windx + boil) * deltat) ** 2) ** (5 / 6) -
                      (xm ** 2 + ym ** 2) ** (5 / 6)) / ro ** (5 / 3)

    # denominator, Zernike sum of squares
    dnm = np.zeros((zern_mx - 1))
    for xx in np.arange(0, zern_mx - 1):
        dnm[xx] = np.sum(np.sum(zern[xx, :, :] ** 2))

    # FFT of all zernikes
    fft_mat = np.zeros((nn + 1, nn + 1, zern_mx - 1)) + 0j
    for jj in np.arange(0, zern_mx - 1):
        fft_mat[:, :, jj] = np.fft.fft2(np.fft.fftshift(zern[jj, :, :]))

    # inner double sum integral
    idsi = np.zeros((zern_mx - 1, zern_mx - 1))
    for xx in np.arange(0, zern_mx - 1):
        for yy in np.arange(0, zern_mx - 1):
            xcorr_fft = np.real(np.fft.fftshift(np.fft.ifft2(fft_mat[:, :, xx] * fft_mat[:, :, yy].conj())))
            idsi[xx, yy] = np.sum(np.sum(xcorr_fft * phs_struct / (dnm[xx] * dnm[yy])))
            # xcorr = signal.correlate2d(zern[xx, :, :], zern[yy, :, :])  # check xcorr results
            # idsi[xx, yy] = np.sum(np.sum(xcorr * phs_struct / (dnm[xx] * dnm[yy])))

    # get n structure function from the phase structure function differences
    phi = la.inv(ch)
    dn = np.zeros(zern_mx-1)
    temp = np.zeros((zern_mx-1, zern_mx-1, zern_mx-1))
    for ii in range(0, zern_mx-1):
        temp[:, :, ii] = np.outer(phi[ii, :], phi[ii, :])
        dn[ii] = np.sum(np.sum(idsi * temp[:, :, ii]))

    # get the n-vector, and correlation functions
    r_0 = 1
    r_n = r_0 - dn/2
    r_n = np.clip(r_n, a_min=0, a_max=None).reshape((1, zern_mx-1))
    n_vec = np.random.normal(size=(1, zern_mx - 1))
    cond_var = 1 - r_n**2
    cond_var = cond_var.reshape((1, zern_mx-1))

    # generate screens from statistics (update frame based on conditional mean and variance)
    atm_lens_phs = np.zeros((nn + 1, nn + 1))
    z_record = np.zeros((zern_mx-1, frames))
    screens = np.zeros((nn+1, nn+1, frames))
    for ii in range(0, frames):
        atm_lens_phs = np.zeros((nn + 1, nn + 1))
        z_scale = ch @ n_vec.T
        z_record[:, ii] = np.squeeze(z_scale)
        for jj in np.arange(0, zern_mx - 1):
            atm_lens_phs = atm_lens_phs + z_scale[jj] * zern[jj, :, :]
        screens[:, :, ii] = atm_lens_phs
        cond_mean = n_vec * r_n
        n_vec = np.sqrt(cond_var) * np.random.normal(size=(zern_mx-1)) + cond_mean

    # check what the screens look like when true
    check_screens_flag = False
    create_animation_flag = False
    if check_screens_flag:
        step = int(frames/4)  # assumes number of frames divisible by 4
        fig, ax = plt.subplots(4, step)
        for ii in range(0, 4):
            for jj in range(0,  step):
                ax[ii, jj].imshow(screens[:, :, jj + ii*4])
                ax[ii, jj].axes.xaxis.set_visible(False)
                ax[ii, jj].axes.yaxis.set_visible(False)
        plt.tight_layout()
        plt.show()

    # create and display animation in HTML
    if create_animation_flag:
        fig = plt.figure()
        def AnimationFunction(f):
            plt.imshow(screens[:, :, f])
        ani = FuncAnimation(fig, AnimationFunction, frames=frames, interval=50)
        writer = PillowWriter(fps=10)
        ani.save('screens.gif', writer=writer)

    print('gen multiple screens')
    return screens


# takes in the zernike polynomials and creates covariance matrix and its cholesky decomp
def generate_zern_polys(zern_mx, nn, d, ro):
    k = 2.2698

    # create zernicke
    zern, idx = zern_poly(zern_mx + 1, nn)
    zern = zern[1: zern_mx, :, :]

    # transfer indices
    n = idx[:, 0]
    m = idx[:, 1]
    p = idx[:, 2]

    # calculate covariance matrix
    covar = np.zeros((zern_mx, zern_mx))
    for xx in np.arange(0, zern_mx):
        for yy in np.arange(0, zern_mx):
            test1 = m[xx] == m[yy]
            test2 = m[xx] == 0
            temp_frac = (p[xx] / 2) / np.ceil(p[xx] / 2)
            p_even = temp_frac == 1
            temp_frac = (p[yy] / 2) / np.ceil(p[yy] / 2)
            p_p_even = temp_frac == 1
            test3 = p_even == p_p_even
            test0 = test2 | test3
            if test1 and test0:
                k_zz = k * np.power(-1, (n[xx] + n[yy] - (2 * m[xx])) / 2) * np.sqrt((n[xx] + 1) * (n[xx] + 1))
                num = k_zz * math.gamma((n[xx] + n[yy] - (5 / 3)) / 2) * np.power(d / ro, 5 / 3)
                dnm = math.gamma((n[xx] - n[yy] + 17 / 3) / 2) * math.gamma((n[yy] - n[xx] + 17 / 3) / 2) * \
                      math.gamma((n[xx] + n[yy] + 23 / 3) / 2)
                covar[xx, yy] = num / dnm

    # factorize covariance matrix using cholesky
    covar = covar[1:zern_mx, 1:zern_mx]
    ch = la.cholesky(covar)

    return zern, ch


# create zernike polynomials from zernike indexes
def zern_poly(i_mx, num_pts):
    del_x = (1 / num_pts) * 2
    x_crd = del_x * np.linspace(int(-num_pts / 2), int(num_pts / 2), int(num_pts + 1))
    xm, ym = np.meshgrid(x_crd, x_crd)
    rm = np.sqrt(xm ** 2 + ym ** 2)
    thm = np.arctan2(ym, xm)

    zern_idx = zern_indexes()

    if i_mx > 1000: print('error: too many zernike polynomials requests')
    zern_idx = zern_idx[0:i_mx - 1, :]

    zern = np.zeros((int(i_mx), int(num_pts + 1), int(num_pts + 1)))
    for ii in np.arange(0, i_mx - 1):
        nn = zern_idx[ii, 0]
        mm = zern_idx[ii, 1]
        if mm == 0:
            zern[ii, :, :] = np.sqrt(nn + 1) * zrf(nn, 0, rm)
        else:
            if np.mod(ii, 2) == 0:
                zern[ii, :, :] = np.sqrt(2 * (nn + 1)) * zrf(nn, mm, rm) * np.cos(mm * thm)
            else:
                zern[ii, :, :] = np.sqrt(2 * (nn + 1)) * zrf(nn, mm, rm) * np.sin(mm * thm)
        mask = (xm ** 2 + ym ** 2) <= 1
        zern[ii] = zern[ii] * mask

    print('zern_poly')
    return zern, zern_idx


# pull zernike indices from csv file, defaults to zern_idx.csv
def zern_indexes(filename: str = 'zern_idx.csv'):
    # read in csv file and convert to list
    raw = csv.DictReader(open(filename))
    raw_list = list(raw)

    # initialize z array
    r = 1000  # number of indixes in zern.csv file
    c = 3  # x, y, i
    z = np.zeros((r, c))

    # get 'x' and 'y' column values as index 'i'
    for row in np.arange(0, r):
        row_vals = raw_list[row]
        z[row, 0] = float(row_vals['x'])
        z[row, 1] = float(row_vals['y'])
        z[row, 2] = float(row_vals['i'])

    return z


# zernike radial function
def zrf(n, m, r):
    rr = 0
    for ii in np.arange(0, (n - m + 1) / 2):
        num = (-1) ** ii * math.factorial(n - ii)
        dnm = math.factorial(ii) * math.factorial(((n + m) / 2) - ii) * math.factorial(((n - m) / 2) - ii)
        rr = rr + (num / dnm) * r ** (n - (2 * ii))
    return rr


# fresnel propagation utilizing fft
def fresnel_prop1(field_s, xms, yms, zo, lam):
    # receive coords
    ds = xms[0, 1] - xms[0, 0]
    ns = np.size(xms, 1) - 1
    dr = 1 / ds / ns
    xr_crd = np.linspace(-(1 / ds) / 2, (1 / ds) / 2, ns + 1)
    xmr, ymr = np.meshgrid(xr_crd, xr_crd)

    # fresnel integral
    quadrtc = np.exp(1j * np.pi * (xms ** 2 + yms ** 2) / (zo * lam))
    field_r = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(quadrtc * field_s)))
    a = np.exp(1j * 2 * np.pi * zo / lam) * np.exp(1j * np.pi * lam * zo * (xmr ** 2 + ymr ** 2)) / (1j * lam * zo)
    field_r = a * field_r

    print("fresnel prop")
    return field_r, xr_crd * (lam * zo)


# fresnel propagation with for loops
def fresnel_prop2(field_s, xms, yms, r_sz, zo, lam, arr_sz):
    # receive coords
    rc = np.arange((-r_sz / 2), (r_sz / 2) + (r_sz * .5 / arr_sz), r_sz / arr_sz) / (lam * zo)
    xmr, ymr = np.meshgrid(rc, rc)

    # fresnel integral
    quadrtc = np.exp(1j * np.pi * (xms ** 2 + yms ** 2) / (zo * lam))
    field_r = np.zeros((arr_sz + 1, arr_sz + 1)) + 0j

    # slow but with correct receive coordinates
    for ii in range(arr_sz):
        for jj in range(arr_sz):
            field_r[ii, jj] = np.sum(
                np.sum(field_s * quadrtc * np.exp(-1j * 2 * np.pi * (xms * rc[jj] + yms * rc[ii]))))

    a = np.exp(1j * 2 * np.pi * zo / lam) * np.exp(1j * np.pi * lam * zo * (xmr ** 2 + ymr ** 2)) / (1j * lam * zo)
    field_r = a * field_r

    print("fresnel prop")
    return field_r, rc * (lam * zo), (r_sz / 2) / arr_sz


# Fraunhofer criteria
def fresnel_criteria(x, y, zeta, eta, lam):
    return np.cbrt(np.pi * ((x - zeta) ** 2 + (y - eta) ** 2) ** 2 / (4 * lam))


# Fresnel criteria
def fraunhofer_criteria(zeta, eta, lam):
    return np.pi * (zeta ** 2 + eta ** 2) / lam


# execute simplified version fraunhofer propagation
def fraunhofer_prop(field_s, lam, zo, xs_crd, ds):
    # x^2+y^2
    sz = np.size(xs_crd, 0)
    dx = 1 / ds / (sz - 1)
    xr_crd = np.linspace(-dx * (sz - 1) / 2, dx * (sz - 1) / 2, sz)
    xm, ym = np.meshgrid(xr_crd, xr_crd)

    a_r = ((xm) ** 2 + (ym) ** 2)

    # amplitude term for the fraunhofer propagation
    a = np.exp(2 * 1j * np.pi * zo / lam) * np.exp(1j * np.pi * lam * zo * a_r) / (1j * lam * zo)

    # fourier transform and do a element-wise multiplication
    fs_fft = (1 / (lam * zo) ** 2) * np.fft.fftshift(np.fft.fft2(field_s))
    field_r = a * fs_fft

    print("fraunhofer prop")
    return field_r, xr_crd * (lam * zo)


# use rayleigh sommerfield propagation on given source field
def rayleigh_sommerfeld_prop(field_s, lam, zo, xs_crd, n_r, sz_r, xc, yc):
    xs_mgrid, ys_mgrid = np.meshgrid(xs_crd, xs_crd)  # meshgrids for source x & y

    # xr_crd = np.arange(-np.floor(sz_r / 2), np.ceil(sz_r / 2)) * dx_r + xc  # array of received x-coordinates
    # yr_crd = np.arange(-np.floor(sz_r / 2), np.ceil(sz_r / 2)) * dx_r + yc  # array of received y-coordinates

    xr_crd = np.arange(-sz_r / 2, (sz_r / 2) + (sz_r / n_r), sz_r / n_r) + xc
    yr_crd = np.arange(-sz_r / 2, (sz_r / 2) + (sz_r / n_r), sz_r / n_r) + yc

    field_r = np.zeros((np.size(xr_crd), np.size(yr_crd))) + 0j  # create receiving field (complex) with zeros

    # perform propagation
    phs_cent_idx = np.ceil(sz_r / 2)
    for ii in range(np.size(yr_crd)):
        for jj in range(np.size(xr_crd)):
            r = np.sqrt((xs_mgrid - xr_crd[jj]) ** 2 + (ys_mgrid - yr_crd[ii]) ** 2 + zo ** 2)
            field_r[ii, jj] = np.sum(
                np.sum(np.multiply(field_s, np.exp(1j * 2 * np.pi * r / lam) / (r ** 2 * lam * 1j))))
            if ii == phs_cent_idx and jj == phs_cent_idx:
                phs_center = 2 * np.pi * r / lam

    print("rs prop")
    return field_r, xr_crd, phs_center


# partition image into four snippets
def partition_image(images, nx, ny, rx, ry, frames):
    xpart = [int((nx/2) - (nx*rx) + 1), int(nx/2), int((nx/2) + (nx*rx))]
    ypart = [int((ny/2) - (ny*ry)), int(ny/2), int((ny/2) + (ny*ry) + 1)]

    temp1 = np.zeros((frames, ypart[1] - ypart[0], xpart[1] - xpart[0]))
    temp2 = np.zeros((frames, ypart[1] - ypart[0], xpart[1] - xpart[0]))
    temp3 = np.zeros((frames, ypart[1] - ypart[0], xpart[1] - xpart[0]))
    temp4 = np.zeros((frames, ypart[1] - ypart[0], xpart[1] - xpart[0]))
    for ii in range(frames):
        temp1[ii, :, :] = images[ii, ypart[1]:ypart[2], xpart[1]:xpart[2]]
        temp2[ii, :, :] = images[ii, ypart[1]:ypart[2], xpart[0]:xpart[1]]
        temp3[ii, :, :] = images[ii, ypart[0]:ypart[1], xpart[1]:xpart[2]]
        temp4[ii, :, :] = images[ii, ypart[0]:ypart[1], xpart[0]:xpart[1]]

    im_parts = np.asarray([temp1, temp2, temp3, temp4])
    print('partitioned image')
    return im_parts


# create animated gif of the data to show the effects of atmosphere over time
def animate_results(field_s1, screens, data, labels, frames):
    fig, ax = plt.subplots(2, 3)
    for ii in range(4):
        data_min = np.min(np.min(data[ii]))
        data_max = np.max(np.max(data[ii]))
        data[ii] = (data[ii] - data_min) / (data_max - data_min) * 255
        data[ii] = data[ii].astype(dtype='uint8')

    def AnimationFunction(f):
        ax[0, 0].imshow(field_s1)
        ax[0, 0].set_title('source')
        ax[0, 0].axis('off')
        ax[0, 1].imshow(data[0][f, :, :])
        ax[0, 1].set_title(f'label:{labels[0, 0]:1.0f} \nlum:{labels[0, 1] * 1e7:0.2f},{labels[0, 2] * 1e7:0.2f}')
        ax[0, 1].axis('off')
        ax[0, 2].imshow(data[1][f, :, :])
        ax[0, 2].set_title(f'label:{labels[1, 0]:1.0f} \nlum:{labels[1, 1] * 1e7:0.2f},{labels[1, 2] * 1e7:0.2f}')
        ax[0, 2].axis('off')
        ax[1, 0].imshow(screens[:, :, f])
        ax[1, 0].set_title('atmos')
        ax[1, 0].axis('off')
        ax[1, 1].imshow(data[2][f, :, :])
        ax[1, 1].set_title(f'label:{labels[2, 0]:1.0f} \nlum:{labels[2, 1] * 1e7:0.2f},{labels[2, 2] * 1e7:0.2f}')
        ax[1, 1].axis('off')
        ax[1, 2].imshow(data[3][f, :, :])
        ax[1, 2].set_title(f'label:{labels[3, 0]:1.0f} \nlum:{labels[3, 1] * 1e7:0.2f},{labels[3, 2] * 1e7:0.2f}')
        ax[1, 2].axis('off')

    ani = FuncAnimation(fig, AnimationFunction, frames=frames, interval=50)
    writer = PillowWriter(fps=10)
    ani.save('light_gen.gif', writer=writer)


# save data to file for later use
def save_file(path, data, labels):
    # convert images to bytes then saves (frames, 140px, 146px)
    for ii in range(4):
        data_min = np.min(np.min(data[ii]))
        data_max = np.max(np.max(data[ii]))
        data[ii] = (data[ii] - data_min) / (data_max - data_min) * 255
        d = data[ii].astype('uint8')
        np.save(file=path+'/'+str(int(labels[ii, 0]))+'/'+str(uuid.uuid4()), arr=d)

    print('saved data')


if __name__ == '__main__':
    # create zernike polynomials once
    zern_mx = 200
    nn = 120  # number of samples along 1D receive plane
    d = 0.28  # diameter of telescope
    ro = 0.07  # fried seeing parameter
    runs = 400

    # create zernike polynomials for use in atmospheric model
    zern, ch = generate_zern_polys(zern_mx, nn, d, ro)

    # loop through the data generation function
    for ii in range(runs):
        light_prop_gen(nn, d, zern_mx, ro, zern, ch)
        print(f'SIM NUM: {ii}')
