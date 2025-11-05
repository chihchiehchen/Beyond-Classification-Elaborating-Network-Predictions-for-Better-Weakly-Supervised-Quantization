For the training data, we use selected cases from [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection)
First we need to extract the group the DICOM slices from their z-position and StudyInstanceUID, we provide the [reoragnized label list](https://drive.google.com/file/d/1QGcGmIYAzSCMC7QJbJnUXTdqWVS92lQT/view?usp=sharing)

For the test data, we test on the [BHSD dataset](https://github.com/White65534/BHSD). For the preprocessing, use [nii_to_png_3ch](https://github.com/chihchiehchen/Beyond-Classification-Elaborating-Network-Predictions-for-Better-Weakly-Supervised-Quantization/blob/main/utils/transfer_to_png.py) to transfer the NIFTI files into our PNG formats. Then iteratively use [demo_code.py](https://github.com/chihchiehchen/Beyond-Classification-Elaborating-Network-Predictions-for-Better-Weakly-Supervised-Quantization/blob/main/demo_code.py) to visualize the generated heatmaps.




