
from utils.visualization_utils import *
from utils.dicom_image import *


def get_region_mask(img_atlas: np.ndarray, id_mask: np.ndarray) -> np.ndarray:
    """ Retrieve mask from the combination of the id regions in the atlas """
    mask = np.zeros_like(img_atlas)

    condition = np.zeros_like(mask, dtype='bool')
    for id in id_mask:
        condition = condition | (img_atlas == id)

    mask[condition] = 1
    return mask


patient = dicom_image(dicom_image_types.FOLDER, "corregistration_data/RM_Brain_3D-SPGR/*.dcm")
plot_median_planes(patient.array_image, patient.pixel_len_mm)

phantom = dicom_image(dicom_image_types.FILE,
                      "corregistration_data/phantom_data/icbm_avg_152_t1_tal_nlin_symmetric_VI.dcm")
plot_median_planes(phantom.array_image, phantom.pixel_len_mm)

atlas = dicom_image(dicom_image_types.FILE, "corregistration_data/phantom_data/AAL3_1mm.dcm")
plot_median_planes(atlas.array_image, atlas.pixel_len_mm, cmap='prism')
create_gif(atlas.array_image, atlas.pixel_len_mm, title="full_map", cmap='prism')

important_regions = np.concatenate((np.array([81, 82]), np.arange(121, 151)), axis=None)
thalamus_mask = get_region_mask(atlas.array_image, important_regions)
plot_median_planes(thalamus_mask, atlas.pixel_len_mm, cmap='tab10')
create_gif(thalamus_mask, atlas.pixel_len_mm, title="thalamus")
