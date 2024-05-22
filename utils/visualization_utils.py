""" Code retrieved and modified from activity_2 and activity_3 branches on https://github.com/PBibiloni/11763 """

import os
import matplotlib
import numpy as np
import scipy
from matplotlib import pyplot as plt, animation
from tqdm import tqdm


def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """ Rotate the image on the axial plane. """

    return scipy.ndimage.rotate(img_dcm, angle_in_degrees, (1, 2), reshape=False, order=0 if (img_dcm.dtype == bool) or (img_dcm.dtype == np.uint8) else 3)

def MIP_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the coronal orientation. """

    return np.max(img_dcm, axis=1)

def MIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the sagittal orientation. """

    return np.max(img_dcm, axis=2)

def MIP_axial_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the sagittal orientation. """

    return np.max(img_dcm, axis=0)


def sagittal_plane(img_dcm: np.ndarray, index = 0) -> np.ndarray:
    """ Compute the sagittal plane of the CT image provided. """

    return img_dcm[:, :, index]    # Why //2?


def coronal_plane(img_dcm: np.ndarray, index = 0) -> np.ndarray:
    """ Compute the coronal plane of the CT image provided. """

    return img_dcm[:, index, :]    # Why //2?


def axial_plane(img_dcm: np.ndarray, index = 0) -> np.ndarray:
    """ Compute the axial plane of the CT image provided. """

    return img_dcm[index, :, :]    # Why //2?


def median_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the median sagittal plane of the CT image provided. """

    return img_dcm[:, :, img_dcm.shape[1]//2]    # Why //2?


def median_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the median sagittal plane of the CT image provided. """

    return img_dcm[:, img_dcm.shape[2]//2, :]


def median_axial_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the median sagittal plane of the CT image provided. """

    return img_dcm[img_dcm.shape[0] // 2, :, :]


def plot_median_planes(img_dcm: np.ndarray, pixel_len_mm: np.ndarray = None, cmap: str = 'bone'):
    # Show MIP/AIP/Median planes
    plt.rcParams["figure.figsize"] = (8, 6)

    fig, ax = plt.subplots(2, 3)
    ax = ax.flatten()
    if pixel_len_mm is not None:
        ax[0].imshow(median_sagittal_plane(img_dcm), cmap=matplotlib.colormaps[cmap],
                     aspect=pixel_len_mm[0] / pixel_len_mm[1])
        ax[0].set_title('Median Sagital')
        ax[1].imshow(median_coronal_plane(img_dcm), cmap=matplotlib.colormaps[cmap],
                     aspect=pixel_len_mm[0] / pixel_len_mm[2])
        ax[1].set_title('Median Coronal')
        ax[2].imshow(median_axial_plane(img_dcm), cmap=matplotlib.colormaps[cmap],
                     aspect=pixel_len_mm[1] / pixel_len_mm[2])
        ax[2].set_title('Median Axial')

        ax[3].imshow(MIP_sagittal_plane(img_dcm), cmap=matplotlib.colormaps[cmap],
                     aspect=pixel_len_mm[0] / pixel_len_mm[1])
        ax[3].set_title('MIP Sagital')
        ax[4].imshow(MIP_coronal_plane(img_dcm), cmap=matplotlib.colormaps[cmap],
                     aspect=pixel_len_mm[0] / pixel_len_mm[2])
        ax[4].set_title('MIP Coronal')
        ax[5].imshow(MIP_axial_plane(img_dcm), cmap=matplotlib.colormaps[cmap],
                     aspect=pixel_len_mm[1] / pixel_len_mm[2])
        ax[5].set_title('MIP Axial')
    else:
        ax[0].imshow(median_sagittal_plane(img_dcm), cmap=matplotlib.colormaps[cmap])
        ax[0].set_title('Median Sagital')
        ax[1].imshow(median_coronal_plane(img_dcm), cmap=matplotlib.colormaps[cmap])
        ax[1].set_title('Median Coronal')
        ax[2].imshow(median_axial_plane(img_dcm), cmap=matplotlib.colormaps[cmap])
        ax[2].set_title('Median Axial')

        ax[3].imshow(MIP_sagittal_plane(img_dcm), cmap=matplotlib.colormaps[cmap])
        ax[3].set_title('MIP Sagital')
        ax[4].imshow(MIP_coronal_plane(img_dcm), cmap=matplotlib.colormaps[cmap])
        ax[4].set_title('MIP Coronal')
        ax[5].imshow(MIP_axial_plane(img_dcm), cmap=matplotlib.colormaps[cmap])
        ax[5].set_title('MIP Axial')

    plt.show()


def plot_planes_arround_point(img_dcm: np.ndarray, point: tuple[int, int, int], pixel_len_mm: np.ndarray = None, cmap: str = 'bone'):
    # Show MIP/AIP/Median planes
    plt.rcParams["figure.figsize"] = (8, 5)

    fig, ax = plt.subplots(1, 3)
    ax = ax.flatten()
    if pixel_len_mm is not None:
        ax[0].imshow(sagittal_plane(img_dcm, point[2]), cmap=matplotlib.colormaps[cmap],
                     aspect=pixel_len_mm[0] / pixel_len_mm[1])
        ax[0].set_title(f'Sagital at {point[2]}')
        ax[1].imshow(coronal_plane(img_dcm, point[1]), cmap=matplotlib.colormaps[cmap],
                     aspect=pixel_len_mm[0] / pixel_len_mm[2])
        ax[1].set_title(f'Coronal at {point[1]}')
        ax[2].imshow(axial_plane(img_dcm, point[0]), cmap=matplotlib.colormaps[cmap],
                     aspect=pixel_len_mm[1] / pixel_len_mm[2])
        ax[2].set_title(f'Axial at {point[0]}')
    else:
        ax[0].imshow(sagittal_plane(img_dcm, point[2]), cmap=matplotlib.colormaps[cmap])
        ax[0].set_title(f'Sagital at {point[2]}')
        ax[1].imshow(coronal_plane(img_dcm, point[1]), cmap=matplotlib.colormaps[cmap])
        ax[1].set_title(f'Coronal at {point[1]}')
        ax[2].imshow(axial_plane(img_dcm, point[0]), cmap=matplotlib.colormaps[cmap])
        ax[2].set_title(f'Axial at {point[0]}')

    plt.show()


def alpha_projection(image: np.ndarray, mask: np.ndarray, image_cmap="bone", mask_cmap="tab10", alpha=0.25) -> np.ndarray:
    C1 = matplotlib.colormaps[image_cmap](image)
    C2 = matplotlib.colormaps[mask_cmap](mask) * mask[..., np.newaxis].astype("bool")

    out = C1
    out[mask > 0] = C1[mask > 0] * (1 - alpha) + C2[mask > 0] * alpha

    return out


def alpha_projection_img(image_1: np.ndarray, image_2: np.ndarray, image_1_cmap="bone", image_2_cmap="bone", alpha=0.25) -> np.ndarray:
    C1 = matplotlib.colormaps[image_1_cmap](image_1)
    C2 = matplotlib.colormaps[image_2_cmap](image_2) * image_2[..., np.newaxis]

    out = C1 * (1 - alpha) + C2 * alpha

    return out


def min_max_norm(image: np.ndarray) -> np.ndarray:
    min_img, max_img = np.min(image), np.max(image)
    normalized = (image - min_img) / (max_img - min_img)
    return normalized


def create_animation(projections: list, cm, folder: str, aspect=None):
    plt.clf()
    fig, ax = plt.subplots()
    ax.set_facecolor("black")

    if aspect is None:
        animation_data = [
            [plt.imshow(img, animated=True, cmap=cm)] for img in projections
        ]
    else:
        animation_data = [
            [plt.imshow(img, animated=True, cmap=cm, aspect=aspect)] for img in projections
        ]
    anim = animation.ArtistAnimation(fig, animation_data,
                                     interval=100, blit=True)
    anim.save(f'results/{folder}/Animation.gif')  # Save animation
    plt.show()  # Show animation


def create_gif(full_ct: np.ndarray, pixel_len_mm: np.ndarray = None, mask: np.ndarray = None, folder: str = "def", cmap: str = "bone"):
    plt.rcParams["figure.figsize"] = (8, 6)
    # Create projections varying the angle of rotation
    #   Configure visualization colormap
    cm = matplotlib.colormaps[cmap]
    #   Configure directory to save results
    os.makedirs(f'results/{folder}/', exist_ok=True)
    #   Create projections
    n = 16
    projections = []

    if (full_ct.dtype != bool) and (full_ct.dtype != np.uint8):
        full_ct = min_max_norm(full_ct)
        full_ct[full_ct < 0.3] = 0  # Remove void noise

    for idx, alpha in tqdm(enumerate(np.linspace(0, 360*(n-1)/n, num=n)), desc="Creating gif", total=n):
        rotated_img = rotate_on_axial_plane(full_ct, alpha)
        projection = MIP_sagittal_plane(rotated_img)
        if mask is not None:
            rotated_mask = rotate_on_axial_plane(mask, alpha)
            mask_projection = MIP_sagittal_plane(rotated_mask)
            projection = alpha_projection(projection, mask_projection)
        else:
            projection = cm(projection) * projection[..., np.newaxis].astype("bool")

        plt.clf()
        ax = plt.axes()
        if pixel_len_mm is None:
            ax.imshow(projection)
        else:
            ax.imshow(projection, aspect=pixel_len_mm[0] / pixel_len_mm[1])
        ax.set_facecolor("black")
        plt.savefig(f'results/{folder}/Projection_{idx}.png')  # Save animation
        projections.append(projection)  # Save for later animation

    # Save and visualize animation
    if pixel_len_mm is None:
        create_animation(projections, cm, folder)
    else:
        create_animation(projections, cm, folder, aspect=pixel_len_mm[0] / pixel_len_mm[1])



def create_gif_correg(full_ct: np.ndarray, pixel_len_mm: np.ndarray = None, image_2: np.ndarray = None, folder: str = "def", cmap: str = "bone", cmap_2: str = "autumn"):
    plt.rcParams["figure.figsize"] = (8, 6)
    # Create projections varying the angle of rotation
    #   Configure visualization colormap
    cm = matplotlib.colormaps[cmap]
    #   Configure directory to save results
    os.makedirs(f'results/{folder}/', exist_ok=True)
    #   Create projections
    n = 16
    projections = []

    if (full_ct.dtype != bool) and (full_ct.dtype != np.uint8):
        full_ct = min_max_norm(full_ct)
        full_ct[full_ct < 0.3] = 0  # Remove void noise

    if (image_2.dtype != bool) and (image_2.dtype != np.uint8):
        image_2 = min_max_norm(image_2)
        image_2[image_2 < 0.3] = 0  # Remove void noise

    for idx, alpha in tqdm(enumerate(np.linspace(0, 360*(n-1)/n, num=n)), desc="Creating gif", total=n):
        rotated_img = rotate_on_axial_plane(full_ct, alpha)
        projection = MIP_sagittal_plane(rotated_img)
        if image_2 is not None:
            rotated_img2 = rotate_on_axial_plane(image_2, alpha)
            img2_projection = MIP_sagittal_plane(rotated_img2)
            projection = alpha_projection_img(projection, img2_projection, cmap, cmap_2)
        else:
            projection = cm(projection) * projection[..., np.newaxis].astype("bool")

        plt.clf()
        ax = plt.axes()
        if pixel_len_mm is None:
            ax.imshow(projection)
        else:
            ax.imshow(projection, aspect=pixel_len_mm[0] / pixel_len_mm[1])
        ax.set_facecolor("black")
        plt.savefig(f'results/{folder}/Projection_{idx}.png')  # Save animation
        projections.append(projection)  # Save for later animation

    # Save and visualize animation
    if pixel_len_mm is None:
        create_animation(projections, cm, folder)
    else:
        create_animation(projections, cm, folder, aspect=pixel_len_mm[0] / pixel_len_mm[1])



def create_gif_sagital(full_ct: np.ndarray, pixel_len_mm: np.ndarray, mask: np.ndarray = None, title: str = "def", cmap: str = "bone"):
    # Create projections varying the angle of rotation
    #   Configure visualization colormap
    cm = matplotlib.colormaps[cmap]
    #   Configure directory to save results
    os.makedirs(f'results/{title}/', exist_ok=True)
    #   Create projections
    n = 32
    projections = []

    if full_ct.dtype != bool:
        full_ct = min_max_norm(full_ct)
        full_ct[full_ct < 0.3] = 0  # Remove void noise

    for idx, s_idx in tqdm(enumerate(np.linspace(0, full_ct.shape[2]-1, num=n, dtype='int')), desc="Creating gif", total=n):
        projection = sagittal_plane(full_ct, s_idx)

        if mask is not None:
            mask_projection = sagittal_plane(mask, s_idx)
            projection = alpha_projection(projection, mask_projection)

        plt.clf()
        ax = plt.axes()
        ax.imshow(projection, cmap=cm, aspect=pixel_len_mm[0] / pixel_len_mm[1])
        ax.set_facecolor("black")
        plt.savefig(f'results/{title}/Projection_{idx}.png')  # Save animation
        projections.append(projection)  # Save for later animation
    # Save and visualize animation
    if pixel_len_mm is None:
        create_animation(projections, cm, title)
    else:
        create_animation(projections, cm, title, aspect=pixel_len_mm[0] / pixel_len_mm[1])


def visualize_coregistration(image_1: np.ndarray, image_2: np.ndarray, cmap_1="bone", cmap_2="autumn", folder:str="coregistration"):
    plt.rcParams["figure.figsize"] = (10, 5)
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(alpha_projection_img(min_max_norm(median_sagittal_plane(image_1)),
                                      min_max_norm(median_sagittal_plane(image_2)), image_1_cmap=cmap_1,
                                      image_2_cmap=cmap_2, alpha=0.5))
    ax[0].set_title('Sagital')
    ax[1].imshow(alpha_projection_img(min_max_norm(median_coronal_plane(image_1)),
                                      min_max_norm(median_coronal_plane(image_2)), image_1_cmap=cmap_1,
                                      image_2_cmap=cmap_2, alpha=0.5))
    ax[1].set_title('Coronal')
    ax[2].imshow(alpha_projection_img(min_max_norm(median_axial_plane(image_1)),
                                      min_max_norm(median_axial_plane(image_2)), image_1_cmap=cmap_1,
                                      image_2_cmap=cmap_2, alpha=0.25))
    ax[2].set_title('Axial')
    plt.savefig(f'results/{folder}.png')
    plt.show()
