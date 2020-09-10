# LSD2

Keras implementation of the method: <br>
*LSD2 -- Joint Denoising and Deblurring of Short and Long Exposure Images with CNNs* [[arXiv](https://arxiv.org/abs/1811.09485)]

## Testing

Download the weights for LSD2 from [here](https://www.dropbox.com/s/q0ujjft2nqlxsln/LSD2_ft.hdf5?dl=0) and put them to the <font color='darkgreen'>*checkpoints*</font> folder. <br> 

To process the images in the folder <font color='darkgreen'>*input/test/...*</font>, execute: <br>
`python predict.py` <br> 

Output images will be saved to <font color='darkgreen'>*output/test/...*</font>. Note that the network expects a pair of short and long exposure images taken with specific camera settings. It will probably not work with other kinds of inputs.

## Training

To train your own model, execute: <br>
`python train.py` <br> 

The folders <font color='darkgreen'>*input/train/...*</font> and <font color='darkgreen'>*input/val/...*</font> contain a few training and validation images. You can replace them with your own.

## Android image acquisition software

Soon to be uploaded.

## Citation

If you find our code helpful in your research or work, please cite our paper.

```
@InProceedings{Mustaniemi_2019_WACV,
  author = {Mustaniemi, Janne and Kannala, Juho and Matas, Jiri and Särkkä, Simo and Heikkilä, Janne},
  title = {LSD_2 - Joint Denoising and Deblurring of Short and Long Exposure Images with CNNs},
  booktitle = {The 31st British Machine Vision Virtual Conference (BMVC)},
  month = {September},
  year = {2020}
}
```
