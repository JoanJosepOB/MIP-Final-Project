# MIP-Final-Project

_Author: Joan Josep Ordóñez Bonet_

This repository contains a solution for the proposed final project on the subject **11763 - Medical Image Processing** on the year 2023-2024 on the **MUSI** at the **UIB**.

To run the repository it is recommended to have a conda environment with `Python>=3.11`, and to install the listed requirements in `requirements.txt`.

The repository is structured as follows:
- In the main folder there are three folders:
  - `utils`: is a package that contains functionalities for loading, visualizing, and applying transformations to DICOM images.
  - `manifest-1714030203846`: contains the data to solve the first part of the project:
    - The folder `1.2.276.0.7230010.3.1.3.8323329.899.1600928677.186044` contains the segmentation data in a single DICOM file.
    - The folder `1.3.6.1.4.1.14519.5.2.1.1706.8374.213776214865122688712708174786` contains the CT data. Each CT slice is contained in a different DICOM file. In total three acquisitions on the same date are stored.
  - `corregistration_data`: contains the data to solve the second part of the project:
    - The folder `RM_Brain_3D-SPGR` contains the patient's CT data by slices in multiple DICOM files.
    - The folder `phantom_data` contains both the normalized phantom image in a single DICOM file `icbm_avg_152_t1_tal_nlin_symmetric_VI.dcm`, and the segmentation of this normalized space in another DICOM file `AAL3_1mm.dcm`, whose labels are found in the txt.
- `DICOM_Vis.py` contains the solution to the first section.
- `3D_Corregis.py` contains the solution to the second section.

To run the solutions each of the python files in the main folder can be directly executed. Any results may appear directly plotted or be saved in a `results` folder in the same execution directory.

## To do
- The implementation of the first part may be tweaked.
- The implementation of the second part of the project is to be further developed.