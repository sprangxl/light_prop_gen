import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.interpolate as itrp
import scipy.linalg as la
import csv
import time


def light_prop_gen():
    start_time = time.time()

    c = 2.9979e8
    h = 6.626e-34
    qe = 1.6022e-19
    kb = 1.38065e-34

    lam = 529e-9  # optimal wavelength of camera (m)
    zo = 35786e3  # geosynch (m)
    x_s1 = 6e3  # edge length of source field (m)
    n_s1 = 30  # number of samples along 1D source plane
    x_r1 = 0.3  # edge length of receive field (m)
    n_r1 = 120  # number of samples along 1D receive plane

    fl = 2.8  # focal length (m)
    lens_d = 0.280  # lense diameter (m)
    outer_d = 0.280  # outer diameter of telescope (m)
    inner_d = 0.095  # inner diameter of mask due to subreflector (m)

    del_t = .1  # exposure time
    intensity_max = 1e-11  # max target intensity (W/m^2)

    lum_max = intensity_max * del_t  # V/m^2 or Ws/m^2

    # create source field
    field_s1, xc_s1 = source_field(n_s1, x_s1, zo, lum_max)

    # propagate source field to front of telescope
    field_r1, xr1_crd, phsc_r1 = rayleigh_sommerfeld_prop(field_s1, lam, zo, xc_s1, n_r1, x_r1, 0, 0)

    # phase screen
    zern_mx = 100  # max number of zernikes used in screen
    ro = 0.1  # seeing parameter
    windx = 1
    windy = 1
    boil = 1
    del_t = .1
    scrn_scale_fac = 0.002

    # add atmospheric phase noise to the recieve plane
    screen, zern_phs = zern_phase_scrn(ro, outer_d, n_r1, zern_mx, xr1_crd, windx, windy, boil, del_t)
    field_r1_atm = field_r1 * np.exp(1j * 2 * np.pi * (lam * c) * scrn_scale_fac * zern_phs)

    # add a mask depicting the telescope front-end
    mask_n = n_r1
    mask_sz = x_r1
    mask1 = circ_mask(mask_n, mask_sz, outer_d, inner_d)
    field_s2 = field_r1_atm * mask1

    # skip over barrel propagation if too slow
    if False:
        # propagate field from front of the telescope to the lense
        z1 = 610e-3  # propagation distance through telescope
        xms2, yms2 = np.meshgrid(xr1_crd, xr1_crd)
        field_r3, xr3_crd, d_r3 = fresnel_prop2(field_s2, xms2, yms2, x_r1, z1, lam, n_r1)
    else:
        # do not add extra step of propagating through telescope barrel
        field_r3 = field_s2
        xr3_crd = xr1_crd
        d_r3 = x_r1 / 2 / n_r1

    # focal plane array (sensor) parameters
    n_sensx = 2448  # number of fpa pixels, x
    n_sensy = 2050  # number of fpa pixels, y
    sens_w = 9.93e-3  # fpa width
    sens_h = 8.7e-3  # fpa height
    pix_sz = 3.45e-6  # pixel size of photodetector

    # focus field based on effective lense focal length, turns fresnel into fraunhofer of distance fl
    mask2 = circ_mask(mask_n, mask_sz, lens_d, 0)
    field_s3 = field_r3 * mask2
    field_r4, xr4_crd = fraunhofer_prop(field_s3, lam, fl, xr3_crd, d_r3)

    # sample region on the ccd array
    sens_crdx = np.linspace(-sens_w / 2, sens_w / 2, n_sensx)
    sens_crdy = np.linspace(-sens_h / 2, sens_h / 2, n_sensy)
    data = sample(field_r4, xr4_crd, sens_crdx, sens_crdy)

    # add detector noise (thermal and photon counting) to the system
    eff = 0.53  # quantum efficiency
    temp = 300  # temperature (K)
    cap = 1e-12  # circuit capacitence (F)
    i_dark = 9.4 * qe

    thermal = np.sqrt(kb * temp * cap / qe ** 2)  # thermal noise
    data_signal = (data * (pix_sz ** 2) * del_t) / (h * c / lam)  # field (V/m^2) to photons
    background = np.max(np.max(data_signal))  # background noise

    thermal_rv = np.random.normal(0, thermal, size=[n_sensy, n_sensx])
    data_signal_rv = np.random.poisson(eff * data_signal)
    background_rv = np.random.poisson(eff * background, size=[n_sensy, n_sensx])*500

    data_noisey = data_signal_rv + thermal_rv + background_rv

    print('time: ', (time.time() - start_time))

    scx_min = np.min(sens_crdx)
    scx_max = np.max(sens_crdx)
    scy_min = np.min(sens_crdy)
    scy_max = np.max(sens_crdy)

    # plot final propagation when true
    if True:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(field_s1)
        ax[0].set(xlabel='original scene')
        ax[1].imshow(data_noisey, extent=[scx_min, scx_max, scy_max, scy_min])
        ax[1].set(xlabel='generated image')
        plt.show()

    print('done')


# space objects desired for detection
def source_field(n, sz, z, lu):
    x = np.arange(-sz / 2, sz / 2 + (sz / n), sz / (n))
    xm, ym = np.meshgrid(x, x)
    field_s = np.zeros((n + 1, n + 1))

    lumin = np.random.uniform(0, lu, size=(2, 2, 2))
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

    return field_s, x


# 11" (28cm) circle with 3.75" (9.5cm) center circle missing
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


# 8MP focal plane array
def sample(field_s, xs_crd, sampx, sampy):
    # determine magnitude of source field
    field_s_mag = np.sqrt(np.real(field_s) ** 2 + np.imag(field_s) ** 2)
    field_rs = itrp.interp2d(xs_crd, xs_crd, field_s_mag, kind='linear')
    # field_rs2 = itrp.RectBivariateSpline(xs_crd, xs_crd, field_s_mag)

    print('resampled')
    return field_rs(sampx, sampy)


def zern_phase_scrn(ro, d, nn, zern_mx, x_crd, windx, windy, boil, deltat):
    k = 2.2698
    a = 6.88

    # create zernicke
    zern, idx = zern_poly(zern_mx + 1, nn)
    zern = zern[1: zern_mx, :, :]

    # transfer indices
    n = idx[:, 0]
    n_p = n
    m = idx[:, 1]
    m_p = m
    p = idx[:, 2]
    p_p = p

    # calculate covariance matrix
    covar = np.zeros((zern_mx, zern_mx))
    for xx in np.arange(0, zern_mx):
        for yy in np.arange(0, zern_mx):
            test1 = m[xx] == m_p[yy]
            test2 = m[xx] == 0
            temp_frac = (p[xx] / 2) / np.ceil(p[xx] / 2)
            p_even = temp_frac == 1
            temp_frac = (p_p[yy] / 2) / np.ceil(p_p[yy] / 2)
            p_p_even = temp_frac == 1
            test3 = p_even == p_p_even
            test0 = test2 | test3
            if test1 and test0:
                k_zz = k * np.power(-1, (n[xx] + n_p[yy] - (2 * m[xx])) / 2) * np.sqrt((n[xx] + 1) * (n_p[xx] + 1))
                num = k_zz * math.gamma((n[xx] + n_p[yy] - (5 / 3)) / 2) * np.power(d / ro, 5 / 3)
                dnm = math.gamma((n[xx] - n_p[yy] + 17 / 3) / 2) * math.gamma((n_p[yy] - n[xx] + 17 / 3) / 2) * \
                      math.gamma((n[xx] + n_p[yy] + 23 / 3) / 2)
                covar[xx, yy] = num / dnm

    # factorize covariance matrix using cholesky
    covar = covar[1:zern_mx, 1:zern_mx]
    ch = la.cholesky(covar)
    rn = np.random.normal(size=(zern_mx - 1, 1))
    z_cf = np.matmul(ch, rn)
    zern_phs = np.zeros((nn + 1, nn + 1))
    for ii in np.arange(0, zern_mx - 1):
        zern_phs = zern_phs + z_cf[ii] * zern[ii, :, :]

    # get phase structure
    xm, ym = np.meshgrid(x_crd, x_crd)
    phs_struct = a * (((ym + (windy + boil) * deltat) ** 2 + (xm + (windx + boil) * deltat) ** 2) ** (5 / 6) -
                      (xm ** 2 + ym ** 2) ** (5 / 6)) / ro ** (5 / 3)

    # inner double sum integral
    fft_mat = np.zeros((nn + 1, nn + 1, zern_mx - 1)) + 0j
    for jj in np.arange(0, zern_mx - 1):
        fft_mat[:, :, jj] = np.fft.fft2(np.fft.fftshift(zern[jj, :, :]))

    # denominator
    dnm = np.zeros((zern_mx - 1))
    for xx in np.arange(0, zern_mx - 1):
        dnm[xx] = np.sum(np.sum(zern[xx, :, :] ** 2))

    # inner double sum integral
    idsi = np.zeros((zern_mx - 1, zern_mx - 1)) + 0j
    for xx in np.arange(0, zern_mx - 1):
        for yy in np.arange(0, zern_mx - 1):
            idsi[xx, yy] = np.sum(
                np.sum(np.fft.fftshift(np.fft.ifft2(fft_mat[:, :, xx] * np.conjugate(fft_mat[:, :, yy])))
                       * phs_struct / (dnm[xx] * dnm[yy])))

    # generate screen
    rn2 = np.random.normal(size=(zern_mx - 1, 1))

    atm_lens_phs2 = np.zeros((nn + 1, nn + 1))
    z_scale = np.matmul(ch, rn2)
    for jj in np.arange(0, zern_mx - 1):
        atm_lens_phs2 = atm_lens_phs2 + (z_scale[jj] * zern[jj, :, :])

    print('zern scrn')
    return atm_lens_phs2, zern_phs


# create zernike polynomials
def zern_poly(i_mx, num_pts):
    del_x = (1 / num_pts) * 2
    x_crd = del_x * np.linspace(int(-num_pts / 2), int(num_pts / 2), int(num_pts + 1))
    xm, ym = np.meshgrid(x_crd, x_crd)
    rm = np.sqrt(xm ** 2 + ym ** 2)
    thm = np.arctan2(ym, xm)

    zern_idx = zern_indexes()
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


# generate indices for creating zernike polynomials
def zern_indexes():
    raw = csv.DictReader(open('zernikes.csv'))
    raw_list = list(raw)
    r = 200
    c = 3
    z = np.zeros((r, c))
    for row in np.arange(0, r):
        row_vals = raw_list[row]
        z[row, 0] = float(row_vals['x'])
        z[row, 1] = float(row_vals['y'])
        z[row, 2] = float(row_vals['i'])
    return z


# zernike radial function
def zrf(n, m, r):
    rr = 0
    for ss in np.arange(0, (n - m + 1) / 2):
        num = (-1) ** ss * math.factorial(n - ss)
        dnm = math.factorial(ss) * math.factorial(((n + m) / 2) - ss) * math.factorial(((n - m) / 2) - ss)
        rr = rr + (num / dnm) * r ** (n - (2 * ss))
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


if __name__ == '__main__':
    light_prop_gen()
