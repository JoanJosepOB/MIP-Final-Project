
from utils.visualization_utils import *
from utils.dicom_image import *

VISUALIZE_SEGMENTATIONS = False


def combine_masks(masks: np.ndarray) -> np.ndarray:
    """ Combine all masks in a single image by doing overwriting in the specified order (mask n over n-1 ... over 1) """
    out = np.zeros_like(masks[0])

    for mask_id, mask in enumerate(masks):
        out[mask > 0] = mask_id + 1

    return out


# Build segmentation image
segmentation = dicom_image(dicom_image_types.FILE,
                           "manifest-1714030203846/HCC-TACE-Seg/HCC_011/1.3.6.1.4.1.14519.5.2.1.1706.8374.631692026948757386108201385429/1.2.276.0.7230010.3.1.3.8323329.899.1600928677.186044/1-1.dcm")


# Divide the 4 segmentations in equally sized images
sliced_segmentation = np.array(np.array_split(segmentation.array_image, len(segmentation.dicom_struct.SegmentSequence)))

for i, segment in enumerate(segmentation.dicom_struct.SegmentSequence):
    sliced_segmentation[i][sliced_segmentation[i] == 1] = segment.SegmentNumber

# Combine all segments in a single mask with each segment with unique IDs
combined_segmentation = combine_masks(sliced_segmentation)

# Create segmentation gifs to check each segmentation
if VISUALIZE_SEGMENTATIONS:
    for i in range(len(sliced_segmentation)):
        create_gif(sliced_segmentation[i], segmentation.pixel_len_mm, title=segmentation.dicom_struct.SegmentSequence[i].SegmentLabel)


# Build CT scan image
ct_image = dicom_image(dicom_image_types.FOLDER, "manifest-1714030203846/HCC-TACE-Seg/HCC_011/1.3.6.1.4.1.14519.5.2.1.1706.8374.631692026948757386108201385429/1.3.6.1.4.1.14519.5.2.1.1706.8374.213776214865122688712708174786/2-*.dcm")

create_gif(ct_image.array_image, ct_image.pixel_len_mm, combined_segmentation, "Full Scan")
