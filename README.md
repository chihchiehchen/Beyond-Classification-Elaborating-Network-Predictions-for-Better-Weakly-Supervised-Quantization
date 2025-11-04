# Beyond-Classification-Elaborating-Network-Predictions-for-Better-Weakly-Supervised-Quantization
This is the official implementation of our work "Beyond Classification: Elaborating Network Predictions for Better Weakly Supervised Quantization"


**Demo Code**

To run the demo code, firstly install the following packages:

```
pip install grad-cam timm
```
Then, run the following script:

```
python demo_code.py --model_checkpoint=<the pretrained checkpoint path> --png_dir=<the image dir you want to visualize the heatmaps> --cam_path=<the dir for saving heatmaps>
```

Checkpoints can be downloaded at https://drive.google.com/file/d/1jy4WIxom5PbhNRL2SzbjVezBCZa7sr_d/view?usp=sharing , where segdense_checkpoint-16.pth.tar is the checkpoint for this model.
(And rdnet_checkpoint-2.pth.tar is the checkpoint for the pretrained RDNet on the ICH dataset)  

--- | ---  
![alt text-1](https://github.com/chihchiehchen/Beyond-Classification-Elaborating-Network-Predictions-for-Better-Weakly-Supervised-Quantization/blob/main/demo_dir/20_ID_661bee514.png =100x100) | ![alt text-2](https://github.com/chihchiehchen/Beyond-Classification-Elaborating-Network-Predictions-for-Better-Weakly-Supervised-Quantization/blob/main/experiments_segrd_segmentation_out/20_ID_661bee514_SAH.png)

To generate heatmaps from your own DICOM/NIfTI files, first we need to transfer the DICOM/NIfTI files into Numpy files, concat three consequent slices together with window width = 130, window level = 25,
and finally save them ans PNG images. We provide the scripts in utils/transfer_to_png.py. Before doing so, you need to install the following packages:  

```
pip install pydicom nibabel
```

to do:
reorganize the training script.
