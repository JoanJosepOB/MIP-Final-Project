import pydicom
import glob
import numpy as np

from enum import Enum

from scipy.spatial.transform import Rotation
from scipy import ndimage
from skimage.transform import resize


class dicom_image_types(Enum):
    FOLDER = 0,
    FILE = 1


class dicom_image:

    def __init__(self, type: dicom_image_types, path: str):
        self.image_type = type

        if type == dicom_image_types.FOLDER:
            files = glob.glob(path)
            self.dicom_struct = [pydicom.dcmread(file) for file in files]
            self.dicom_struct.sort(key=lambda slice: slice.SliceLocation, reverse=True)
            acquisition_number = self.dicom_struct[0].AcquisitionNumber

            for slice in self.dicom_struct:
                same_acquisition = (acquisition_number == slice.AcquisitionNumber)
                if not same_acquisition:
                    print("WARNING: Selected images come from different acquisitions.")
                    break

            self.array_image = np.array([struct_slice.pixel_array for struct_slice in self.dicom_struct])
            patient_orientation = self.dicom_struct[0].ImageOrientationPatient
            try:
                overlap_ratio = self.dicom_struct[0].SpacingBetweenSlices / self.dicom_struct[0].SliceThickness
                if overlap_ratio < 1.0:
                    print(f"Slices overlap by {overlap_ratio*100}%")
                elif overlap_ratio > 1.0:
                    print(f"Empty data between slices of {(overlap_ratio - 1)*100}%")

                # Consider each slice to be smaller/larger than its thickness, to remove overlap or fill gaps without generating new slices
                self.pixel_len_mm = [self.dicom_struct[0].SliceThickness*overlap_ratio, self.dicom_struct[0].PixelSpacing[0], self.dicom_struct[0].PixelSpacing[1]]  # Pixel length in mm [z, y, x]
            except Exception as ex:
                # Spacing between slices not found
                self.pixel_len_mm = [self.dicom_struct[0].SliceThickness, self.dicom_struct[0].PixelSpacing[0], self.dicom_struct[0].PixelSpacing[1]]  # Pixel length in mm [z, y, x]

        elif type == dicom_image_types.FILE:
            self.dicom_struct = pydicom.dcmread(path)

            self.array_image = self.dicom_struct.pixel_array
            patient_orientation = self.dicom_struct.SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient

            dimension_params = self.dicom_struct.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
            try:
                self.pixel_len_mm = np.array([dimension_params.SliceThickness, dimension_params.PixelSpacing[0], dimension_params.PixelSpacing[1]])  # Pixel length in mm [z, y, x]
            except Exception as error:
                print("No Slice Thickness specified, defaulting to 1")
                self.pixel_len_mm = np.array([1, dimension_params.PixelSpacing[0], dimension_params.PixelSpacing[1]])  # Pixel length in mm [z, y, x]
        else:
            print("Invalid dicom image type:", type)
            return

        # Flip inverted axis so all are positive: X=[1, 0, 0], Y=[0, 1, 0]
        r1 = np.array(patient_orientation[:3])
        r2 = np.array(patient_orientation[3:])
        r3 = np.cross(r1, r2)
        T_matrix = np.array([r1, r2, r3]).T

        rot_angles = Rotation.from_matrix(T_matrix).as_euler("XYZ", degrees=True)
        if np.abs(rot_angles[0]) > 0:
            self.array_image = ndimage.rotate(self.array_image, rot_angles[0], (1, 2))
        if np.abs(rot_angles[1]) > 0:
            self.array_image = ndimage.rotate(self.array_image, rot_angles[1], (0, 2))
        if np.abs(rot_angles[2]) > 0:
            self.array_image = ndimage.rotate(self.array_image, rot_angles[2], (0, 1))


    def normalized(self):
        return resize(self.array_image, (np.round(self.pixel_len_mm[0] * self.array_image.shape[0]),
                                                     np.round(self.pixel_len_mm[1] * self.array_image.shape[1]),
                                                     np.round(self.pixel_len_mm[2] * self.array_image.shape[2])))

    def to_same_resolution(self, image: np.ndarray):
        return resize(image, (np.round(image.shape[0] / self.pixel_len_mm[0]),
                                          np.round(image.shape[1] / self.pixel_len_mm[1]),
                                          np.round(image.shape[2] / self.pixel_len_mm[2])), order=0 if image.dtype == bool else None)

    def slice_positions(self):
        coords = []

        if self.image_type == dicom_image_types.FILE:
            group_sequence = self.dicom_struct.PerFrameFunctionalGroupsSequence
            for slice_data in group_sequence:
                coords.append(slice_data.PlanePositionSequence[0].ImagePositionPatient)
        elif self.image_type == dicom_image_types.FOLDER:
            group_sequence = self.dicom_struct
            for slice_data in group_sequence:
                coords.append(slice_data.ImagePositionPatient)

        return coords
