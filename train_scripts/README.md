For the training data, we use selected cases from [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection)

For the test data, we test on the [BHSD dataset](https://github.com/White65534/BHSD). For the preprocessing, use [nii_to_png_3ch](https://github.com/chihchiehchen/Beyond-Classification-Elaborating-Network-Predictions-for-Better-Weakly-Supervised-Quantization/blob/main/utils/transfer_to_png.py) to transfer the NIFTI files into our PNG formats. Then iteratively use [demo_code.py](https://github.com/chihchiehchen/Beyond-Classification-Elaborating-Network-Predictions-for-Better-Weakly-Supervised-Quantization/blob/main/demo_code.py) to visualize the generated heatmaps.




