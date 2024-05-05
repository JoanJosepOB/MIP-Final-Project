import pydicom
import glob
import numpy as np

from enum import Enum


class dicom_image_types(Enum):
    FOLDER = 0,
    FILE = 1


class dicom_image:

    def __init__(self, type: dicom_image_types, path: str):

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

            self.pixel_len_mm = [self.dicom_struct[0].SliceThickness, self.dicom_struct[0].PixelSpacing[0], self.dicom_struct[0].PixelSpacing[1]]  # Pixel length in mm [z, y, x]

        elif type == dicom_image_types.FILE:
            self.dicom_struct = pydicom.dcmread(path)

            self.array_image = self.dicom_struct.pixel_array
            patient_orientation = self.dicom_struct.SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient

            dimension_params = self.dicom_struct.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
            self.pixel_len_mm = np.array([1, dimension_params.PixelSpacing[0], dimension_params.PixelSpacing[1]])  # Pixel length in mm [z, y, x]
        else:
            print("Invalid dicom image type:", type)
            return

        # Flip inverted axis so all are positive: X=[1, 0, 0], Y=[0, 1, 0]
        reverse_x, reverse_y = patient_orientation[0] < 0, patient_orientation[4] < 0
        if reverse_x:
            self.array_image = np.flip(self.array_image, axis=0)
        if reverse_y:
            self.array_image = np.flip(self.array_image, axis=1)
