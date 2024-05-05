""" Code retrieved and modified from activity_2 and activity_3 branches on https://github.com/PBibiloni/11763 """

import os
import matplotlib
import numpy as np
import scipy
from matplotlib import pyplot as plt, animation
from tqdm import tqdm


def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """ Rotate the image on the axial plane. """

    return scipy.ndimage.rotate(img_dcm, angle_in_degrees, (1, 2), reshape=False)

def MIP_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the coronal orientation. """

    return np.max(img_dcm, axis=1)

def MIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the sagittal orientation. """

    return np.max(img_dcm, axis=2)


def median_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the median sagittal plane of the CT image provided. """

    return img_dcm[:, :, img_dcm.shape[1]//2]    # Why //2?


def median_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the median sagittal plane of the CT image provided. """

    return img_dcm[:, img_dcm.shape[2]//2, :]


def plot_median_planes(img_dcm: np.ndarray, pixel_len_mm: np.ndarray, cmap: str = 'bone'):
    # Show MIP/AIP/Median planes
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(median_sagittal_plane(img_dcm), cmap=matplotlib.colormaps[cmap],
                 aspect=pixel_len_mm[0] / pixel_len_mm[1])
    ax[0].set_title('Median')
    ax[1].imshow(MIP_sagittal_plane(img_dcm), cmap=matplotlib.colormaps[cmap],
                 aspect=pixel_len_mm[0] / pixel_len_mm[1])
    ax[1].set_title('MIP')
    fig.suptitle('Sagittal')
    plt.show()
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(median_coronal_plane(img_dcm), cmap=matplotlib.colormaps[cmap],
                 aspect=pixel_len_mm[0] / pixel_len_mm[1])
    ax[0].set_title('Median')
    ax[1].imshow(MIP_coronal_plane(img_dcm), cmap=matplotlib.colormaps[cmap],
                 aspect=pixel_len_mm[0] / pixel_len_mm[1])
    ax[1].set_title('MIP')
    fig.suptitle('Coronal')
    plt.show()


def alpha_projection(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    C1 = matplotlib.colormaps["bone"](image)
    C2 = matplotlib.colormaps["tab10"](mask) * mask[..., np.newaxis].astype("bool")

    alpha = 0.5
    out = C1 * (1 - alpha) + C2 * alpha

    return out


def create_gif(full_ct: np.ndarray, pixel_len_mm: np.ndarray, mask: np.ndarray = None, title: str = "def", cmap: str = "bone"):
    # Create projections varying the angle of rotation
    #   Configure visualization colormap
    img_min = np.amin(full_ct)
    img_max = np.amax(full_ct)
    cm = matplotlib.colormaps[cmap]
    fig, ax = plt.subplots()
    #   Configure directory to save results
    os.makedirs(f'results/{title}/', exist_ok=True)
    #   Create projections
    n = 16
    projections = []

    for idx, alpha in tqdm(enumerate(np.linspace(0, 360*(n-1)/n, num=n)), desc="Creating gif", total=n):
        rotated_img = rotate_on_axial_plane(full_ct, alpha)
        projection = MIP_sagittal_plane(rotated_img)
        if mask is not None:
            rotated_mask = rotate_on_axial_plane(mask, alpha)
            mask_projection = MIP_sagittal_plane(rotated_mask)

            projection = alpha_projection(projection, mask_projection)

        plt.clf()
        plt.imshow(projection, cmap=cm, vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0] / pixel_len_mm[1])
        plt.savefig(f'results/{title}/Projection_{idx}.png')      # Save animation
        projections.append(projection)  # Save for later animation
    # Save and visualize animation
    plt.clf()
    animation_data = [
        [plt.imshow(img, animated=True, cmap=cm, vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0] / pixel_len_mm[1])]
        for img in projections
    ]
    anim = animation.ArtistAnimation(fig, animation_data,
                              interval=100, blit=True)
    anim.save(f'results/{title}/Animation.gif')  # Save animation
    plt.show()                              # Show animation