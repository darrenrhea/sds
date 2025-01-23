# https://github.com/handong1587/PSF_generation/tree/master

import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image

def create_trajectory(traj_size=64, anxiety=None, centripetal= None, gaussian_term=None, freq_big_shakes=0, num_t=2000, max_total_length=60, do_show=False):
    if anxiety is None:
        anxiety = 0.05 * random.random()

    if centripetal is None:
        centripetal = 1.0 * random.random()
    
    if gaussian_term is None:
        gaussian_term = 0.7 * random.random()
    
    if freq_big_shakes is None:
        freq_big_shakes = 0.2 * random.random()

    init_angle = 360 * random.random()
    v0 = np.exp(1j * np.deg2rad(init_angle))
    v = v0 * max_total_length / (num_t - 1)

    if anxiety > 0:
        v = v0 * anxiety

    x = np.zeros(num_t, dtype=complex)
    tot_length = 0
    abrupt_shakes_counter = 0

    for t in range(num_t - 1):
        if random.random() < freq_big_shakes * anxiety:
            next_direction = 2 * v * np.exp(1j * (np.pi + (random.random() - 0.5)))
            abrupt_shakes_counter += 1
        else:
            next_direction = 0

        dv = next_direction + anxiety * (gaussian_term * (np.random.randn() + 1j * np.random.randn()) - centripetal * x[t]) * (max_total_length / (num_t - 1))
        v = v + dv
        v = (v / abs(v)) * max_total_length / (num_t - 1)
        x[t + 1] = x[t] + v
        tot_length += abs(x[t + 1] - x[t])

    x -= 1j * np.min(np.imag(x)) + np.min(np.real(x))
    x -= 1j * (np.imag(x[0]) % 1) + (np.real(x[0]) % 1) - 1 - 1j
    x += 1j * np.ceil((traj_size - np.max(np.imag(x))) / 2) + np.ceil((traj_size - np.max(np.real(x))) / 2)

    if do_show:
        plt.figure()
        plt.plot(x.real, x.imag)
        plt.plot(x[0].real, x[0].imag, 'rx')
        plt.plot(x[-1].real, x[-1].imag, 'ro')
        plt.axis([0, traj_size, 0, traj_size])
        plt.legend(['Traj Curve', 'init', 'end'])
        plt.title(f'anxiety: {anxiety}, number of abrupt shakes: {abrupt_shakes_counter}')
        plt.show()

    traj_curve = {
        'x': x,
        'TotLength': tot_length,
        'Anxiety': anxiety,
        'MaxTotalLength': max_total_length,
        'nAbruptShakes': abrupt_shakes_counter
    }

    return traj_curve


def create_psfs(traj_curve, psf_size=63, T=[1/1000, 1/10, 1/2, 1], do_show=False, do_center=True):
    if isinstance(psf_size, int):
        psf_size = [psf_size, psf_size]

    psf_number = len(T)
    num_t = len(traj_curve['x'])
    x = traj_curve['x']

    if do_center:
        x = x - np.mean(x) + (psf_size[1] + 1 + 1j * (psf_size[0] + 1)) / 2

    psfs = []

    triangle_fun = lambda d: np.maximum(0, (1 - np.abs(d)))
    triangle_fun_prod = lambda d1, d2: triangle_fun(d1) * triangle_fun(d2)

    for jj in range(len(T)):
        prev_t = T[jj - 1] if jj > 0 else 0
        psf = np.zeros(psf_size)

        for t in range(num_t):
            t_proportion = 0
            if prev_t * num_t < t < T[jj] * num_t:
                t_proportion = 1
            elif prev_t * num_t < t - 1 < T[jj] * num_t:
                t_proportion = T[jj] * num_t - (t - 1)
            elif prev_t * num_t < t < T[jj] * num_t:
                t_proportion = t - prev_t * num_t
            elif prev_t * num_t < t - 1 < T[jj] * num_t:
                t_proportion = (T[jj] - prev_t) * num_t

            m2 = min(psf_size[1] - 1, max(1, int(np.floor(x[t].real))))
            M2 = m2 + 1
            m1 = min(psf_size[0] - 1, max(1, int(np.floor(x[t].imag))))
            M1 = m1 + 1

            psf[m1, m2] += t_proportion * triangle_fun_prod(x[t].real - m2, x[t].imag - m1)
            psf[m1, M2] += t_proportion * triangle_fun_prod(x[t].real - M2, x[t].imag - m1)
            psf[M1, m2] += t_proportion * triangle_fun_prod(x[t].real - m2, x[t].imag - M1)
            psf[M1, M2] += t_proportion * triangle_fun_prod(x[t].real - M2, x[t].imag - M1)

        psfs.append(psf / num_t)

    if do_show:
        C, D = [], []
        for jj in range(len(T)):
            C.append(psfs[jj])
            D.append(psfs[jj] / np.max(psfs[jj]))

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(np.concatenate(C, axis=0), cmap='hot', aspect='auto')
        plt.title('All PSF normalized w.r.t. the same maximum')
        plt.subplot(1, 2, 2)
        plt.imshow(np.concatenate(D, axis=0), cmap='hot', aspect='auto')
        plt.title('Each PSF normalized w.r.t. its own maximum')
        plt.show()

    return psfs



def create_blurred_raw_color(y, psf, lam, sigma_gauss=0, init=None):
    if init is not None:
        np.random.seed(init)
        random.seed(init)

    # Rescale the original image
    y = y * lam

    # plt.figure()
    # plt.imshow(y / lam)
    # plt.title(f'yhat')
    # plt.show()

    yN, xN, channels = y.shape
    ghy, ghx, _ = psf.shape

    # Generate Blurred Observation
    # Pad PSF with zeros to the whole image domain, and center it
    big_v = np.zeros((yN, xN, channels))
    y0 = yN // 2 - round((ghy - 1) / 2)
    x0 = xN // 2 - round((ghx - 1) / 2)
    big_v[y0:y0 + ghy, x0:x0 + ghx, :] = psf


    # Frequency response of the PSF
    V = np.fft.fft2(big_v)
    
    # Perform blurring (convolution is obtained by product in frequency domain)
    y_blur = np.real(np.fft.ifft2(V * np.fft.fft2(y)))

    plt.matshow(y_blur[:, :, 0])
    plt.colorbar()
    plt.show()

    # Add noise terms
    # Poisson Noise (signal and time dependent)
    Raw = np.random.poisson(lam=np.clip(y_blur, 0, None), size=y_blur.shape)

    # Gaussian Noise (signal and image independent)
    Raw = Raw + sigma_gauss * np.random.randn(*Raw.shape)
    
    return Raw, np.real(V)

def load_and_process_image(file_path):
    """
    Load an image and convert it to a NumPy array of doubles.
    """
    img = Image.open(file_path)
    return np.asarray(img, dtype=np.float64) / 255.0





# Example usage
if __name__ == '__main__':
    traj_curve = create_trajectory(do_show=True)
    psfs = create_psfs(traj_curve, T=[0.5], do_show=True, do_center=True)

    # img = load_and_process_image('20200725PIT-STL-CFCAM-PITCHCAST_inning1_000600.jpg')

    # # Generate the sequence of motion blurred observations
    # padded_image = []

    # for ii in range(len(psfs)):
    #     psf_3 = np.repeat(psfs[ii][:, :, np.newaxis], 3, axis=2)

    #     plt.matshow(psf_3[:, :, 0])
    #     plt.title(f'psf_3')
    #     plt.show()


    #     plt.figure()
    #     plt.imshow(img)
    #     plt.title(f'img')
    #     plt.show()

    #     z = create_blurred_raw_color(img, psf_3, 2048, 0.05)

    #     plt.figure()
    #     plt.imshow(z[0] / np.max(z[0]))
    #     plt.title(f'Image having exposure time {ii}')
    #     plt.show()

    #     im_temp = z[0] / np.max(z[0])
    #     im_temp[:psf_3.shape[0], :psf_3.shape[1], :psf_3.shape[2]] = psf_3 / np.max(psf_3)
    #     padded_image.append(im_temp)

    # # Display and save the sequence of observations
    # final_image = np.concatenate(padded_image, axis=1)
    # plt.figure()
    # plt.imshow(final_image)
    # plt.title('Sequence of observations, PSFs is shown in the upper left corner')
    # plt.show()

    # #final_image_pil = Image.fromarray((final_image * 255).astype(np.uint8))
    # #final_image_pil.save('blu.jpg')