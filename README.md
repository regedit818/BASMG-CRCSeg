## **Boundary-Aware Spectral and Morphological Guidance Method for Feature-Driven Colorectal Cancer Segmentation **

------

This is the official code of Boundary-Aware Spectral and Morphological Guidance Method for Feature-Driven Colorectal Cancer Segmentation (TMI 2026).

------

> Abstract—Precise segmentation of medical images plays a crucial role in modern clinical practice, providing important foundations for the quantitative analysis of medical images and clinical decision making. However, although deep learning techniques have achieved significant success in conventional medical image segmentation, they still exhibit obvious limitations when faced with complex structure segmentation tasks such as colorectal cancer: high-quality medical data acquisition is not only difficult, but also even when data is relatively sufficient, the diverse morphological features of lesions cannot be adequately represented due to their high variability, which severely limits the generalization capability of traditional data-driven segmentation methods. To address these challenge, we propose a feature-driven segmentation model that improves performance by deeply mining intrinsic data information rather than relying on parameter stacking.  Specifically, we introduce a spectrum–prior–boundary triple modeling paradigm, where frequency domain reconstruction and modulation is employed to establish mappings between different frequency bands and heterogeneous signals to identify ambiguous signals, a level set–based segmentation algorithm is used to construct abdominal anatomical distance fields to incorporate morphological priors, and an auxiliary edge branch is designed by integrating deep semantic and shallow detail features to strengthen boundary awareness. Extensive experiments on colorectal cancer segmentation demonstrate that the proposed method achieves significant improvements in both segmentation accuracy and boundary perception over other state-of-the-art methods. Furthermore, evaluations on lung cancer and breast cancer segmentation tasks validate its strong generalization capability across diverse lesion types.

------

![1](./img/1.jpg)

## Architecture overview of 

The overall framework of our model. The 3D data are first fed into the encoder, which consists of L encoder blocks, where each block takes the output of the previous one as its input. The extracted features are then downsampled and passed into the decoder, which consists of L-1 decoder blocks. Each decoder block takes as input three components: the output of the previous decoder block, the output of the corresponding encoder block, and the level set energy function downsampled to the same scale. During decoding, the outputs of the shallowest decoder block 1 and the deepest decoder block L-1 are jointly fed into the HED module to obtain the modulated shallow features and the edge prediction for the auxiliary task. Additionally, UP CONV denotes an upsampling convolution used to reduce the feature size, while DOWN CONV consists of a transposed convolution to restore the feature size followed by a standard convolution. (a) Frequency domain reconstruction module. (b) LSF-guided grouped skip module. (c) Frequency domain Reconstruction Module.



------



## How to Use

- Download and configure [**nnUNet**](https://github.com/MIC-DKFZ/nnUNet)

- Move **BASMG** and **nnUNetTrainer_FFTMixShift_UDEHead_LSFMFMGsplit.py** to **.../nnUNet/nnunetv2/training/nnUNetTrainer/** of the configured nnUNet
- Use **BASMG** just like nnUNet:





# The GitHub repository is under construction......
