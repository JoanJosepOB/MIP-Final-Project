
from utils.visualization_utils import *
from utils.dicom_image import *


def dict_to_mask(mask: dict, out_shape: tuple[int, int, int]) -> np.ndarray:
    mask_array = np.zeros(out_shape, dtype='uint8')

    for i in mask.keys():
        mask_array[i - 1, :, :] = mask[i]

    return mask_array.astype('bool')


def combine_masks(masks: dict, out_shape: tuple[int, int, int]) -> np.ndarray:
    """ Combine all masks in a single image by doing overwriting in the specified order (mask n over n-1 ... over 1) """

    out = np.zeros(out_shape, dtype='uint8')

    for mask_id, mask in masks.items():
        mask_array = dict_to_mask(mask, out_shape)
        out[mask_array > 0] = mask_id

    return out


# Build CT scan image
ct_image = dicom_image(dicom_image_types.FOLDER, "manifest-1714030203846/HCC-TACE-Seg/HCC_011/1.3.6.1.4.1.14519.5.2.1.1706.8374.631692026948757386108201385429/1.3.6.1.4.1.14519.5.2.1.1706.8374.213776214865122688712708174786/2-*.dcm")

# Build segmentation image
segmentation = dicom_image(dicom_image_types.FILE,
                           "manifest-1714030203846/HCC-TACE-Seg/HCC_011/1.3.6.1.4.1.14519.5.2.1.1706.8374.631692026948757386108201385429/1.2.276.0.7230010.3.1.3.8323329.899.1600928677.186044/1-1.dcm")

# Divide the 4 segmentations in equally sized images
segments = {segment.SegmentNumber: {i+1: None for i in range(ct_image.array_image.shape[0])} for segment in segmentation.dicom_struct.SegmentSequence}

for segment_info, segment_slice in zip(segmentation.dicom_struct.PerFrameFunctionalGroupsSequence, segmentation.array_image):
    dimension_values = segment_info.FrameContentSequence[0].DimensionIndexValues
    segments[dimension_values[0]][dimension_values[1]] = segment_slice

# Combine all segments in a single mask with each segment with unique IDs
combined_segmentation = combine_masks(segments, ct_image.array_image.shape)

# Create segmentation gifs to check each segmentation
create_gif(combined_segmentation, ct_image.pixel_len_mm, cmap='tab10', folder="Segmentation/Segments")

for i, k in enumerate(segments.keys()):
    create_gif(ct_image.array_image, ct_image.pixel_len_mm, dict_to_mask(segments[k], ct_image.array_image.shape), folder="Segmentation/" + segmentation.dicom_struct.SegmentSequence[i].SegmentLabel)

create_gif(ct_image.array_image, ct_image.pixel_len_mm, combined_segmentation, "Segmentation/Full Scan")
