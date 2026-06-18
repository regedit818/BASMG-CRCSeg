import copy

from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from threadpoolctl import threadpool_limits
from torch.nn import functional as F
import numpy as np
import torch
from typing import Tuple, Union, List, Dict, Any
from torch import nn, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.training.nnUNetTrainer.BASMG.unet_FFTMixShift_UDEHead_LSFMFMGsplit import M_UNet

from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from torch import distributed as dist
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.collate_outputs import collate_outputs
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, save_json, join
from time import time, sleep
from typing import Tuple, Union, List

import numpy as np
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform

from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


class nnUNetTrainer_FFTMixShift_UDEHead_LSFMFMGsplit(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 5e-3
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 1000
        self.current_epoch = 0
        self.enable_deep_supervision = True
        self.laplace_edge = Laplace3DEdgeDetector(threshold=0.5)
        self.edge_loss = BondaryCELoss()
        self.logger.my_fantastic_logging['edge_losses'] = list()
        self.logger.my_fantastic_logging['val_edge_losses'] = list()
        self.egde_a = 0.5

    # 前三项为类信息，由configuration_manager获取
    # 第四项为输入通道
    # 第五项为输出通道，也就是类别
    # 第六项为是否开启深监督
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
                       use_k=0.1,
                       direction=True,
                       spacing=(3.2999961376190186, 0.5077999830245972, 0.5077999830245972),
                       deep_supervision=enable_deep_supervision
                       )

        return model

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)

            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    # def _do_i_compile(self):
    #     return False

    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            # 这里完全可以整个进度条吧
            train_outputs = []
            train_outputs2 = []
            pbar = tqdm(total=self.num_iterations_per_epoch, dynamic_ncols=False, ncols=100)
            for batch_id in range(self.num_iterations_per_epoch):
                # 使用迭代器传递样本，并反向传播
                y1, y2 = self.train_step(next(self.dataloader_train))
                train_outputs.append(y1)
                train_outputs2.append(y2)
                # 自定义进度条
                pbar.set_description("[Training]")
                pbar.set_postfix({"batch_id": batch_id + 1})
                pbar.update(1)
            # 记得这个
            pbar.close()
            # 计算平均loss
            self.on_train_epoch_end(train_outputs, train_outputs2)

            # 无梯度，进行验证
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                val_outputs2 = []
                pbar = tqdm(total=self.num_val_iterations_per_epoch, dynamic_ncols=False, ncols=100)
                for batch_id in range(self.num_val_iterations_per_epoch):
                    y1, y2 = self.validation_step(next(self.dataloader_val))
                    val_outputs.append(y1)
                    val_outputs2.append(y2)
                    # 自定义进度条
                    pbar.set_description("[val]")
                    pbar.set_postfix({"batch_id": batch_id + 1})
                    pbar.update(1)
                # 记得这个
                pbar.close()
                self.on_validation_epoch_end(val_outputs, val_outputs2)

            self.on_epoch_end()

        self.on_train_end()

    def train_step(self, batch: dict) -> tuple[dict[str, Any], dict[str, Any]]:
        data = batch['data']
        level = batch['level']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        level = level.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # 计算edge不需要反向传播
        with torch.no_grad():
            gt_edge = self.laplace_edge(target[0])

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output, edge = self.network(data, True, level)
            # del data
            l = self.loss(output, target)
            el = self.edge_loss(edge, gt_edge)
            # 只保存dice_ce_loss
            tl = l + self.egde_a * el

        if self.grad_scaler is not None:
            self.grad_scaler.scale(tl).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            tl.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}, {'loss': el.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> tuple[dict[str, Any], dict[str, Any]]:
        data = batch['data']
        target = batch['target']
        gt_edge = self.laplace_edge(target[0].cuda())

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output, edge = self.network(data, True)
            # del data
            l = self.loss(output, target)
            el = self.edge_loss(edge, gt_edge)
            del data

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return ({'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard},
                {'loss': el.detach().cpu().numpy()})

    def on_train_epoch_end(self, train_outputs1: List[dict], train_outputs2: List[dict]):
        outputs = collate_outputs(train_outputs1)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])

        self.logger.log('train_losses', loss_here, self.current_epoch)

        #################################################################
        outputs = collate_outputs(train_outputs2)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])

        self.logger.log('edge_losses', loss_here, self.current_epoch)

    def on_validation_epoch_end(self, val_outputs: List[dict], val_outputs2: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

        # 边缘损失
        outputs = collate_outputs(val_outputs2)

        if self.is_ddp:
            losses_val = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_val, outputs['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs['loss'])

        self.logger.log('val_edge_losses', loss_here, self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss',
                               np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('edge_loss',
                               np.round(self.logger.my_fantastic_logging['edge_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('val_edge_losses',
                               np.round(self.logger.my_fantastic_logging['val_edge_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:

        transforms1 = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms1.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms1.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms1.append(Convert2DTo3DTransform())

        transforms2 = []
        # 水平集分割不应该做这些
        transforms2.append(MyRandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms2.append(MyRandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))

        transforms3 = []
        transforms3.append(MyRandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms3.append(MyRandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms3.append(MyRandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms3.append(MyRandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms3.append(MyRandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms3.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms3.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms3.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms3.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms3.append(
                MyRandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms3.append(
                MyRandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms3.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms3.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return DualTransformWrapper(MyComposeTransforms(transforms1),
                                    MyComposeTransforms(transforms2),
                                    MyComposeTransforms(transforms3))

    def get_dataloaders(self):
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?

        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, transforms=tr_transforms)
            dl_val = nnUNetDataLoader2D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, transforms=val_transforms)
        else:
            dl_tr = nnUNetDataLoaderLevelSet3D(dataset_tr, self.batch_size,
                                               initial_patch_size,
                                               self.configuration_manager.patch_size,
                                               self.label_manager,
                                               oversample_foreground_percent=self.oversample_foreground_percent,
                                               sampling_probabilities=None, pad_sides=None, transforms=tr_transforms)
            dl_val = nnUNetDataLoader3D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, transforms=val_transforms)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val


class nnUNetTrainer_FFTMixShift_UDEHead_LSFMFMGsplit_CRC(nnUNetTrainer_FFTMixShift_UDEHead_LSFMFMGsplit):
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
                       use_k=0.1,
                       direction=True,
                       spacing=(3.2999961376190186, 0.5077999830245972, 0.5077999830245972),
                       deep_supervision=enable_deep_supervision
                       )
        return model


class nnUNetTrainer_FFTMixShift_UDEHead_LSFMFMGsplit_ATLAS(nnUNetTrainer_FFTMixShift_UDEHead_LSFMFMGsplit):
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
                       spacing=(2.999998450279236, 1.09375, 1.09375),
                       deep_supervision=enable_deep_supervision
                       )
        return model


class nnUNetTrainer_FFTMixShift_UDEHead_LSFMFMGsplit_ISPY1(nnUNetTrainer_FFTMixShift_UDEHead_LSFMFMGsplit):
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
                       use_k=0.9,
                       direction=False,
                       spacing=(1.0, 1.0, 1.0),
                       deep_supervision=enable_deep_supervision
                       )
        return model


class nnUNetTrainer_FFTMixShift_UDEHead_LSFMFMGsplit_PanSegData(nnUNetTrainer_FFTMixShift_UDEHead_LSFMFMGsplit):
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


#############################################################################################################
class nnUNetDataLoaderLevelSet3D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)
            case_properties.append(properties)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_bbox_ubs = np.minimum(shape, bbox_ubs)

            # At this point you might ask yourself why we would treat nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple(
                [slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            padding = ((0, 0), *padding)
            data_all[j] = np.pad(data, padding, 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, padding, 'constant', constant_values=-1)

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    level = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['original']['image'])
                        level.append(tmp['levelset']['image'])
                        segs.append(tmp['original']['segmentation'])
                    data_all = torch.stack(images)
                    level_all = torch.stack(level)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images

            return {'data': data_all, 'target': seg_all, 'keys': selected_keys, 'level': level_all}

        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}


class MyRandomTransform(BasicTransform):
    def __init__(self, transform: BasicTransform, apply_probability: float = 1):
        super().__init__()
        self.transform = transform
        self.apply_probability = apply_probability

    def get_parameters(self, **data_dict) -> dict:
        return {"apply_transform": torch.rand(1).item() < self.apply_probability}

    def apply(self, data_dict: dict, **params) -> dict:
        if params['apply_transform']:
            if len(params) < 2:
                params = self.transform.get_parameters(**data_dict)
            return self.transform.apply(data_dict, **params), params
        else:
            return data_dict, {}

    def __repr__(self):
        ret_str = f"{type(self).__name__}(p={self.apply_probability}, transform={self.transform})"
        return ret_str


class MyComposeTransforms(BasicTransform):
    def __init__(self, transforms: List[BasicTransform]):
        super().__init__()
        self.transforms = transforms

    def apply(self, data_dict, params):
        if len(params) == 0:
            params_list = []
            for t in self.transforms:
                now_params = t.get_parameters(**data_dict)
                if t.__class__.__name__ == 'MyRandomTransform':
                    data_dict, RandomParams = t.apply(data_dict, **now_params)
                    now_params.update(RandomParams)
                else:
                    data_dict = t.apply(data_dict, **now_params)
                params_list.append(now_params)
            return data_dict, params_list
        else:
            for t, p in zip(self.transforms, params):
                # 其实[1]就是应用过的参数了啊,为什么有点丑
                # 简单脱壳
                while not type(data_dict) == dict:
                    data_dict = data_dict[0]
                data_dict = t.apply(data_dict, **p)
            return data_dict, params


class DualTransformWrapper(BasicTransform):
    def __init__(self, transforms1: MyComposeTransforms, transforms2: MyComposeTransforms,
                 transforms3: MyComposeTransforms):
        super().__init__()
        self.transforms1 = transforms1
        self.transforms2 = transforms2  # 2是噪声和高斯模糊
        self.transforms3 = transforms3

    def apply(self, sample: dict, **params):
        step1, _ = self.transforms1.apply(copy.deepcopy(sample), [])
        step2, _ = self.transforms2.apply(copy.deepcopy(step1), [])

        # self._save_random_states()
        step3_1, copy_params = self.transforms3.apply(step1, [])
        # self._restore_random_states()
        step3_2, _ = self.transforms3.apply(step2, copy_params)

        return {
            'original': step3_2,
            'levelset': step3_1
        }


# 下面是拉普拉斯核
class Laplace3DEdgeDetector(nn.Module):
    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold

        self.laplace_kernel = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1,
            bias=False
        )
        kernel = torch.tensor([
            [[[0, 1, 0],
              [1, -4, 1],
              [0, 1, 0]]]
        ], dtype=torch.float32).cuda()
        self.laplace_kernel.weight = nn.Parameter(kernel)

        self.dilation_kernel = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1,
            bias=False
        )
        dilation_kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32).cuda()
        self.dilation_kernel.weight = nn.Parameter(dilation_kernel, requires_grad=False)

    def forward(self, x):
        x = x.float()
        B, C, D, H, W = x.shape
        x = x.view(B * D, 1, H, W)

        edge1 = torch.abs(self.laplace_kernel(x))
        # edge1 = self.dilation_kernel(edge1)
        edge1 = (edge1 > self.threshold).float()

        x[x > 0] = 1
        edge = torch.abs(self.laplace_kernel(x))
        edge = self.dilation_kernel(edge)
        edge = (edge > self.threshold).float()
        # edge = edge * x

        edge = edge + edge1
        edge[edge > 0] = 1

        edge = edge.view(B, 1, D, H, W)
        return edge


class BondaryCELoss(nn.Module):
    def __init__(self):
        super(BondaryCELoss, self).__init__()

    def forward(self, bd_pre, target):
        target_background = 1.0 - target
        target = torch.cat([target_background, target], dim=1)

        log_p = bd_pre.permute(0, 2, 3, 4, 1).contiguous().view(1, -1)
        target_t = target.view(1, -1)

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)

        weight = torch.zeros_like(log_p).to(torch.float32)
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num / sum_num
        weight[neg_index] = pos_num / sum_num

        # 花里胡哨,其实都是0.5,等效于除类别进行归一化

        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

        return loss


if __name__ == '__main__':
    pre = torch.randn((1, 2, 32, 224, 256))
    gt = torch.bernoulli(torch.full((1, 1, 32, 224, 256), 0.5))

    loss = BondaryCELoss()
    print(loss(pre, gt))
