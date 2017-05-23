# Keras VDSR
An implementation of the Super Resolution CNN proposed in:

Kim, Jiwon, Jung Kwon Lee, and Kyoung Mu Lee. "Accurate image super-resolution using very deep convolutional networks." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.

Written in Python using Keras on top of TensorFlow

- [The author's project page](http://cv.snu.ac.kr/research/VDSR/)
- To download the required data for training/testing, please refer to the README.md at data directory.

## Files
- vdsr.py : main training file
- test_vdsr.py : test VDSR on an arbitrary image

## Usage
To download the required data for training/testing, please refer to the README.md at data directory.

The training images come from the 291 images data set. Simply run the "vdsr.py" script to begin training. At the end, the model weights will be saved. To test the model, you can use 'vdsr.py'.

This also supports multiple scales with the global variables TRAIN_SCALES and VALID_SCALES