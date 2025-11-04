# Beyond-Classification-Elaborating-Network-Predictions-for-Better-Weakly-Supervised-Quantization
This is the official implementation of our work "Beyond Classification: Elaborating Network Predictions for Better Weakly Supervised Quantization"


**\Demo Code**

To run the demo code, firstly install the following packages:

```
pip install grad-cam, timm
```
Then, run the following script:

```
python demo_code.py --model_checkpoint=<the pretrained checkpoint path> --png_dir=<the image dir you want to visualize the heatmaps> --cam_path=<the dir for saving heatmaps>
```

Checkpoints can be downloaded at https://drive.google.com/file/d/1jy4WIxom5PbhNRL2SzbjVezBCZa7sr_d/view?usp=sharing , where segdense_checkpoint-16.pth.tar is the checkpoint for this model.
(And rdnet_checkpoint-2.pth.tar is the checkpoint for the pretrained RDNet on the ICH dataset)

to Do:
reorganize the training script.
