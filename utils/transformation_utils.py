""" Code retrieved and modified from activity_5 and activity_6 branches on https://github.com/PBibiloni/11763 """

import numpy as np
import quaternion
import math

from scipy.optimize import minimize


def min_max_norm(image: np.ndarray) -> np.ndarray:
    min_img, max_img = np.min(image), np.max(image)
    normalized = (image - min_img) / (max_img - min_img)
    return normalized


def apply_transform(reference: np.ndarray, input: np.ndarray, params):
    coords = get_all_cell_coords(reference)
    return apply_quaternion_transform(input, coords, params)


def apply_inv_transform(reference: np.ndarray, input: np.ndarray, params):
    coords = get_all_cell_coords(input)
    return apply_inv_quaternion_transform(reference, coords, params)


def translation_then_axial_rotation(coords: np.ndarray, parameters: tuple[float, ...]):
    t1, t2, t3, angle_in_rads, v1, v2, v3 = parameters
    # Normalize axis of rotation to avoid restrictions on optimizer
    v_norm = math.sqrt(sum([coord ** 2 for coord in [v1, v2, v3]]))
    v1, v2, v3 = v1 / v_norm, v2 / v_norm, v3 / v_norm

    trans_coords = coords + np.array([t1, t2, t3])

    rot_axis = np.array([v1, v2, v3]) * angle_in_rads
    rot_quaternion = quaternion.from_rotation_vector(rot_axis)

    final_coords = quaternion.rotate_vectors(rot_quaternion, trans_coords, axis=1)

    return final_coords


def translation_then_axial_rotation_inv(coords: np.ndarray, parameters: tuple[float, ...]):
    t1, t2, t3, angle_in_rads, v1, v2, v3 = parameters
    # Normalize axis of rotation to avoid restrictions on optimizer
    v_norm = math.sqrt(sum([coord ** 2 for coord in [v1, v2, v3]]))
    v1, v2, v3 = v1 / v_norm, v2 / v_norm, v3 / v_norm

    rot_axis = np.array([v1, v2, v3]) * -angle_in_rads
    rot_quaternion = quaternion.from_rotation_vector(rot_axis)

    rot_coords = quaternion.rotate_vectors(rot_quaternion, coords, axis=1)

    final_coords = rot_coords - np.array([t1, t2, t3])

    return final_coords


def apply_inv_quaternion_transform(input: np.ndarray, ref_lookup_coords: np.ndarray, params):
    # Apply transformation with respect to center of img
    correction_to_orig = np.array([input.shape[0] // 2, input.shape[1] // 2, input.shape[2] // 2])
    final_coords = translation_then_axial_rotation_inv(ref_lookup_coords - correction_to_orig, params)
    input_lookup_coords = np.round(final_coords).astype(int) + correction_to_orig

    result = np.full_like(input, 0.0)

    keep_coords = filter_coords(input_lookup_coords, input.shape)
    ref_fill = ref_lookup_coords[keep_coords]
    inp_read = input_lookup_coords[keep_coords]
    result[ref_fill[:, 0], ref_fill[:, 1], ref_fill[:, 2]] = input[inp_read[:, 0], inp_read[:, 1], inp_read[:, 2]]

    return result


def filter_coords(coords: np.ndarray, bounds: tuple[int, int, int]):
    return (0 <= coords[:, 0]) & (coords[:, 0] < bounds[0]) & (0 <= coords[:, 1]) & (coords[:, 1] < bounds[1]) & (0 <= coords[:, 2]) & (coords[:, 2] < bounds[2])


def apply_quaternion_transform(input: np.ndarray, ref_lookup_coords: np.ndarray, params):
    # Apply transformation with respect to center of img
    correction_to_orig = np.array([input.shape[0] // 2, input.shape[1] // 2, input.shape[2] // 2])
    final_coords = translation_then_axial_rotation(ref_lookup_coords - correction_to_orig, params)
    input_lookup_coords = np.round(final_coords).astype(int) + correction_to_orig

    result = np.full_like(input, 0.0)

    keep_coords = filter_coords(input_lookup_coords, input.shape)
    ref_fill = ref_lookup_coords[keep_coords]
    inp_read = input_lookup_coords[keep_coords]
    result[ref_fill[:, 0], ref_fill[:, 1], ref_fill[:, 2]] = input[inp_read[:, 0], inp_read[:, 1], inp_read[:, 2]]

    return result


def get_all_cell_coords(img: np.ndarray):
    X_axis = np.arange(0, img.shape[0])
    Y_axis = np.arange(0, img.shape[1])
    Z_axis = np.arange(0, img.shape[2])
    coords_matrix = []
    for X in X_axis:
        for Y in Y_axis:
            for Z in Z_axis:
                coords_matrix.append(np.array([X, Y, Z]))

    return np.array(coords_matrix)


def coregister_images(ref_image: np.ndarray, inp_image: np.ndarray, get_param_history = False):
    """ Coregister two sets of landmarks using a rigid transformation. """

    initial_parameters = [
        0, 0, 0,  # Translation
        0,  # Angle
        1, 0, 0  # Rotation axis
    ]

    parameter_history = {"Error": [], "Tx": [], "Ty": [], "Tz": [], "Angle": [], "Vx": [], "Vy": [], "Vz": []}
    norm_ref_image = min_max_norm(ref_image)
    norm_ref_image[norm_ref_image < 0.3] = 0.0  # Remove void noise
    norm_inp_image = min_max_norm(inp_image)
    norm_inp_image[norm_inp_image < 0.3] = 0.0  # Remove void noise

    coords_matrix = get_all_cell_coords(norm_ref_image)

    def function_to_minimize(parameters):
        """ Transform input coordinates, then compare with reference."""

        t1, t2, t3, angle_in_rads, v1, v2, v3 = parameters
        parameter_history["Tx"].append(t1)
        parameter_history["Ty"].append(t2)
        parameter_history["Tz"].append(t3)
        parameter_history["Angle"].append(angle_in_rads)
        parameter_history["Vx"].append(v1)
        parameter_history["Vy"].append(v2)
        parameter_history["Vz"].append(v3)

        result = apply_quaternion_transform(norm_inp_image, coords_matrix, parameters)
        resid = np.mean(np.square(norm_ref_image - result).flatten())

        parameter_history["Error"].append(resid)

        return resid


    # Apply scalar function optimization
    result = minimize(function_to_minimize, x0=np.array(initial_parameters), method='Powell')

    if get_param_history:
        return result, parameter_history

    return result
