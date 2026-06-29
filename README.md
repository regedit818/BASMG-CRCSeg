## **Boundary-Aware Spectral and Morphological Guidance Method for Feature-Driven Colorectal Cancer Segmentation**

This is the official code of  Boundary-Aware Spectral and Morphological Guidance Method for Feature-Driven Colorectal Cancer Segmentation ([TMI 2026](https://ieeexplore.ieee.org/abstract/document/11573336)).

------

> Abstract—Precise segmentation of medical images plays a crucial role in modern clinical practice, providing important foundations for the quantitative analysis of medical images and clinical decision making. However, although deep learning techniques have achieved significant success in conventional medical image segmentation, they still exhibit obvious limitations when faced with complex structure segmentation tasks such as colorectal cancer: high-quality medical data acquisition is not only difficult, but also even when data is relatively sufficient, the diverse morphological features of lesions cannot be adequately represented due to their high variability, which severely limits the generalization capability of traditional data-driven segmentation methods. To address these challenge, we propose a feature-driven segmentation model that improves performance by deeply mining intrinsic data information rather than relying on parameter stacking.  Specifically, we introduce a spectrum–prior–boundary triple modeling paradigm, where frequency domain reconstruction and modulation is employed to establish mappings between different frequency bands and heterogeneous signals to identify ambiguous signals, a level set–based segmentation algorithm is used to construct abdominal anatomical distance fields to incorporate morphological priors, and an auxiliary edge branch is designed by integrating deep semantic and shallow detail features to strengthen boundary awareness. Extensive experiments on colorectal cancer segmentation demonstrate that the proposed method achieves significant improvements in both segmentation accuracy and boundary perception over other state-of-the-art methods. Furthermore, evaluations on lung cancer and breast cancer segmentation tasks validate its strong generalization capability across diverse lesion types.

------

![1](./img/1.jpg)

## Architecture overview of BASMG

The overall framework of our model. The 3D data are first fed into the encoder, which consists of L encoder blocks, where each block takes the output of the previous one as its input. The extracted features are then downsampled and passed into the decoder, which consists of L-1 decoder blocks. Each decoder block takes as input three components: the output of the previous decoder block, the output of the corresponding encoder block, and the level set energy function downsampled to the same scale. During decoding, the outputs of the shallowest decoder block 1 and the deepest decoder block L-1 are jointly fed into the HED module to obtain the modulated shallow features and the edge prediction for the auxiliary task. Additionally, UP CONV denotes an upsampling convolution used to reduce the feature size, while DOWN CONV consists of a transposed convolution to restore the feature size followed by a standard convolution. (a) Frequency domain reconstruction module. (b) LSF-guided grouped skip module. (c) Frequency domain Reconstruction Module.

------

## How to Use

- Download and configure [**nnUNet**](https://github.com/MIC-DKFZ/nnUNet)

- Move **BASMG** and **nnUNetTrainer_FFTMixShift_UDEHead_LSFMFMGsplit.py** to **.../nnUNet/nnunetv2/training/nnUNetTrainer/** of the configured nnUNet

- Use **BASMG** just like nnUNet:

  

> ### Data Preprocessing

```
nnUNetv2_plan_and_preprocess -d 300 --verify_dataset_integrity
```

We conducted extensive experiments on benchmarks: CRC dataset, [ATLAS](https://atlas-challenge.u-bourgogne.fr/dataset), [ISPY1](https://www.cancerimagingarchive.net/analysis-result/ispy1-tumor-seg-radiomics/) and [PanSegData](https://osf.io/kysnj/).



> ### Training

```
nnUNetv2_train 100 3d_fullres 0 -tr nUNetTrainer_FFTMixShift_UDEHead_LSFMFMGsplit_CRC

nnUNetv2_train 200 3d_fullres 0 -tr nnUNetTrainer_FFTMixShift_UDEHead_LSFMFMGsplit_ATLAS

nnUNetv2_train 300 3d_fullres 0 -tr nnUNetTrainer_FFTMixShift_UDEHead_LSFMFMGsplit_ISPY1

nnUNetv2_train 400 3d_fullres 0 -tr nnUNetTrainer_FFTMixShift_UDEHead_LSFMFMGsplit_PanSegData
```

If you intend to use your own dataset, please overload nnUNetTrainer_FFTMixShift_UDEHead_LSFMFMGsplit and configure the **three parameters (use_k, direction, spacing)** of the network architecture. Such as:

    class nnUNetTrainer_FFTMixShift_UDEHead_LSFMFMGsplit_YourData(nnUNetTrainer_FFTMixShift_UDEHead_LSFMFMGsplit):
        def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                     device: torch.device = torch.device('cuda')):
            """used for debugging plans etc"""
            super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
    
        @staticmethod
        def build_network_architecture(architecture_class_name: str,
                                       arch_init_kwargs: dict,
                                       arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                       num_input_channels: int,
                                       num_output_channels: int,
                                       enable_deep_supervision: bool = True) -> nn.Module:
            len_num = 15
            model = M_UNet(num_input_channels,
                           num_output_channels,
                           arch_init_kwargs['features_per_stage'][:len_num],
                           arch_init_kwargs['kernel_sizes'][:len_num],
                           arch_init_kwargs['strides'][:len_num],
                           use_k=0.8,
                           direction=False,
                           spacing=(4.399994373321533, 1.09375, 1.09375),
                           deep_supervision=enable_deep_supervision
                           )
            return model

**use_k**: threshold for the level set.  
**direction**: direction of the level set, True for forward.  
**spacing**: spacing obtained from nnUNet processing.



> ### Testing

```
nnUNetv2_predict -i .../nnUNetFrame/nnUNet_raw/Dataset100_your_dataset/imagesTs/ -o .../your_predict_path/ -d 100 -c 3d_fullres -tr nnUNetTrainer_FFTMixShift_UDEHead_LSFMFMGsplit_YourData
```

​                           

## Acknowledgement

This repository is built based on [nnUNet](https://github.com/MIC-DKFZ/nnUNet) repository.
