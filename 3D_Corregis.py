
from utils.visualization_utils import *
from utils.dicom_image import *

from skimage.transform import resize

from utils.transformation_utils import coregister_images, apply_transform, apply_inv_transform


def get_region_mask(img_atlas: np.ndarray, id_mask: np.ndarray) -> np.ndarray:
    """ Retrieve mask from the combination of the id regions in the atlas """
    mask = np.zeros_like(img_atlas)

    condition = np.zeros_like(mask, dtype='bool')
    for id in id_mask:
        condition = condition | (img_atlas == id)

    mask[condition] = 1
    return mask.astype(bool)


def adjust_crop(inp: np.ndarray, ref: np.ndarray):
    """ Adjusts the input size to match the ref size. Note that inp must be smaller than ref in all 3 axis"""

    margins = [np.abs(inp.shape[0] - ref.shape[0]) // 2,
               np.abs(inp.shape[1] - ref.shape[1]) // 2,
               np.abs(inp.shape[2] - ref.shape[2]) // 2]
    spares = [np.abs(inp.shape[0] - ref.shape[0]) % 2,
              np.abs(inp.shape[1] - ref.shape[1]) % 2,
              np.abs(inp.shape[2] - ref.shape[2]) % 2]

    return inp[margins[0]:-(margins[0] + spares[0]), margins[1]:-(margins[1] + spares[1]), margins[2]:-(margins[2] + spares[2])]


# Load all required data
patient = dicom_image(dicom_image_types.FOLDER, "./corregistration_data/RM_Brain_3D-SPGR/*.dcm")
phantom = dicom_image(dicom_image_types.FILE, "./corregistration_data/phantom_data/icbm_avg_152_t1_tal_nlin_symmetric_VI.dcm")
atlas = dicom_image(dicom_image_types.FILE, "./corregistration_data/phantom_data/AAL3_1mm.dcm")

# Adjust corregistration image sizes
resized_phantom = adjust_crop(phantom.normalized(), atlas.normalized())
resized_patient = adjust_crop(patient.normalized(), resized_phantom)

# Get smaller versions of the images to speed up the process
patient_resized = resize(resized_patient, (resized_patient.shape[0] // 3, resized_patient.shape[1] // 3, resized_patient.shape[2] // 3))
phantom_resized = resize(resized_phantom, (resized_patient.shape[0] // 3, resized_patient.shape[1] // 3, resized_patient.shape[2] // 3))

if os.path.isfile("./results/Coregistration/corregistration_params.npy"):
    print("Loading corregistration params...")
    final_params = np.load("./results/Coregistration/corregistration_params.npy")
else:
    print("Applying corregistration...")
    result, param_history = coregister_images(phantom_resized, patient_resized, get_param_history=True)
    final_params = result.x

    with open('./results/Coregistration/corregistration_params.npy', 'wb') as f:
        np.save(f, result.x)

    fig, ax = plt.subplots(3,3)
    ax = ax.flatten()
    for i, param in enumerate(param_history.keys()):
        ax[i].plot(np.arange(len(param_history[param])), param_history[param])
        ax[i].set_title(param)
    plt.show()

t1, t2, t3, angle_in_rads, v1, v2, v3 = final_params
print(f'Parameters found:')
print(f'  >> Translation: ({t1}, {t2}, {t3}).')
print(f'  >> Rotation angle (rad): {angle_in_rads}')
print(f'  >> Rotation axis: ({v1}, {v2}, {v3})')

transformed_patient = apply_transform(resized_phantom, resized_patient, final_params)
transformed_phantom = apply_inv_transform(resized_phantom, resized_patient, final_params)

visualize_coregistration(transformed_patient, resized_phantom, title="Normalized Space")
visualize_coregistration(transformed_phantom, resized_patient, title="Patient Space")

# Extract the thalamus region
important_regions = np.concatenate((np.array([81, 82]), np.arange(121, 151)), axis=None)
thalamus_mask = get_region_mask(atlas.array_image, important_regions)

# Convert thalamus to patient space and visualize it
patient_thalamus = patient.to_same_resolution(thalamus_mask)
adj_patient = adjust_crop(patient.array_image, patient_thalamus)
transformed_thalamus = apply_inv_transform(patient_thalamus, adj_patient, final_params)
create_gif(adj_patient, mask=transformed_thalamus, pixel_len_mm=patient.pixel_len_mm, title="Patient Thalamus", cmap="bone")
