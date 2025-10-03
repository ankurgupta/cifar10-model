# CIFAR 10 - Model : 85% Accuracy on 150K paramaeters

## Summary

- This model is created on CIFAR 10 dataset
- It has got **85%** accuracy with **150K** paramaeters
- Used **Albumentation** library for image Augmentations.
- _Horizonatal Flip_, _shiftScaleRotate_ , _coarseDropout_ is being used from Albumentation library.
- _Strided_ Convolution is uded instead of Maxpooling whenever required.
- **Dilated convolution** is used in the layer before the GAP layer to increase the Receptive field
- **Depthwise Separable Convolution** used in the intial convolution block.
- Final Global Receptive field is : 93 with the help of dilated convolution

- **Training and Validation Logs** attached here for reference.[[**Training_Logs**]](./training_logs.md)
- You can find the Colab Notebook here. [[**Notebook**]](./CIFAR10_model_1.ipynb)

## Model Details

- 4 Convolution Block used
- Structure : C1 ➡ C2 ➡ C3 ➡ C4 ➡ Output
- C1, C2, C3, C4 are convolution blocks and O is the output block
- Each block has varied number of layers from 1 to 4.
- C1 blcok will have **Depthwise Separable Convolution** layer.
- C4 block will have **Dilated Convolution** layer.
- Albumentation library used for _Horizonatal Flip_, _shiftScaleRotate_ , _coarseDropout_
- GAP layer and 1x1 convolution is used in the output layer instead of FC layer.
- **Parameters** - 150,576 number of trainable paramters

```
cuda
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
       BatchNorm2d-2           [-1, 16, 32, 32]              32
              ReLU-3           [-1, 16, 32, 32]               0
           Dropout-4           [-1, 16, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]           4,608
       BatchNorm2d-6           [-1, 32, 32, 32]              64
              ReLU-7           [-1, 32, 32, 32]               0
           Dropout-8           [-1, 32, 32, 32]               0
            Conv2d-9           [-1, 32, 32, 32]             320
           Conv2d-10           [-1, 64, 32, 32]           2,112
      BatchNorm2d-11           [-1, 64, 32, 32]             128
             ReLU-12           [-1, 64, 32, 32]               0
           Conv2d-13           [-1, 64, 16, 16]          36,864
      BatchNorm2d-14           [-1, 64, 16, 16]             128
             ReLU-15           [-1, 64, 16, 16]               0
          Dropout-16           [-1, 64, 16, 16]               0
           Conv2d-17           [-1, 16, 16, 16]           1,024
      BatchNorm2d-18           [-1, 16, 16, 16]              32
             ReLU-19           [-1, 16, 16, 16]               0
           Conv2d-20           [-1, 32, 16, 16]           4,608
      BatchNorm2d-21           [-1, 32, 16, 16]              64
             ReLU-22           [-1, 32, 16, 16]               0
          Dropout-23           [-1, 32, 16, 16]               0
           Conv2d-24           [-1, 32, 16, 16]           9,216
      BatchNorm2d-25           [-1, 32, 16, 16]              64
             ReLU-26           [-1, 32, 16, 16]               0
          Dropout-27           [-1, 32, 16, 16]               0
           Conv2d-28             [-1, 64, 8, 8]          18,432
      BatchNorm2d-29             [-1, 64, 8, 8]             128
             ReLU-30             [-1, 64, 8, 8]               0
          Dropout-31             [-1, 64, 8, 8]               0
           Conv2d-32             [-1, 16, 8, 8]           1,024
      BatchNorm2d-33             [-1, 16, 8, 8]              32
             ReLU-34             [-1, 16, 8, 8]               0
           Conv2d-35             [-1, 32, 8, 8]           4,608
      BatchNorm2d-36             [-1, 32, 8, 8]              64
             ReLU-37             [-1, 32, 8, 8]               0
          Dropout-38             [-1, 32, 8, 8]               0
           Conv2d-39             [-1, 64, 8, 8]          18,432
      BatchNorm2d-40             [-1, 64, 8, 8]             128
             ReLU-41             [-1, 64, 8, 8]               0
          Dropout-42             [-1, 64, 8, 8]               0
           Conv2d-43             [-1, 64, 4, 4]          36,864
      BatchNorm2d-44             [-1, 64, 4, 4]             128
             ReLU-45             [-1, 64, 4, 4]               0
          Dropout-46             [-1, 64, 4, 4]               0
           Conv2d-47             [-1, 16, 4, 4]           1,024
      BatchNorm2d-48             [-1, 16, 4, 4]              32
             ReLU-49             [-1, 16, 4, 4]               0
           Conv2d-50             [-1, 64, 4, 4]           9,216
      BatchNorm2d-51             [-1, 64, 4, 4]             128
             ReLU-52             [-1, 64, 4, 4]               0
AdaptiveAvgPool2d-53             [-1, 64, 1, 1]               0
           Conv2d-54             [-1, 10, 1, 1]             640
================================================================
Total params: 150,576
Trainable params: 150,576
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 4.74
Params size (MB): 0.57
Estimated Total Size (MB): 5.33
----------------------------------------------------------------
```
