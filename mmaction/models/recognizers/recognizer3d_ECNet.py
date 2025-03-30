# Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
from torch import Tensor

from mmengine.model import merge_dict
from mmengine.runner.checkpoint import load_checkpoint
from mmaction.registry import MODELS
from mmaction.utils import OptSampleList
from .base import BaseRecognizer

from mmengine.logging import MMLogger
logger = MMLogger.get_current_instance()

@MODELS.register_module()
class Recognizer3DECNet(BaseRecognizer):
    """3D recognizer model framework."""
    def __init__(self, backbone, cls_head = None, neck = None, train_cfg = None, test_cfg = None, data_preprocessor = None, \
                 edge_model=None, edge_feat_idx=None, distill_loss=None, checkpoints=None):
        super().__init__(backbone, cls_head, neck, train_cfg, test_cfg, data_preprocessor)

        assert edge_model is not None, "Please set config for edge model"
        assert os.path.exists(checkpoints['cloud_ckpt']), f"Invalid cloud model checkpoint path {checkpoints['cloud_ckpt']}"
        assert os.path.exists(checkpoints['edge_ckpt']), f"Invalid edge model checkpoint path {checkpoints['edge_ckpt']}"

        # init edge model
        self.edge_model = MODELS.build(edge_model)
        self.edge_feat_idx = edge_feat_idx
        # self.edge_cls_head = MODELS.build(edge_cls_head)
        # self.edge_neck = MODELS.build(edge_neck)

        # init self distillation loss
        self.distill_loss = MODELS.build(distill_loss)

        # 
        self.checkpoints, self.ckpts_loaded = checkpoints, False

    def load_stage1_ckpts(self):
        # edge model
        logger.info(f"Loading pretrained edge checkpoint form {self.checkpoints['edge_ckpt']}")
        load_checkpoint(
            self,
            self.checkpoints['edge_ckpt'],
            map_location="cpu",
            strict=False,
        )
        # cloud model
        logger.info(f"Loading pretrained edge checkpoint form {self.checkpoints['cloud_ckpt']}")
        load_checkpoint(
            self.edge_model,
            self.checkpoints['cloud_ckpt'],
            map_location="cpu",
            strict=False,
        )
        self.ckpts_loaded = True

    def extract_feat(self,
                     inputs: Tensor,
                     stage: str = 'neck',
                     data_samples: OptSampleList = None,
                     test_mode: bool = False) -> tuple:
        """Extract features of different stages.

        Args:
            inputs (torch.Tensor): The input data.
            stage (str): Which stage to output the feature.
                Defaults to ``'neck'``.
            data_samples (list[:obj:`ActionDataSample`], optional): Action data
                samples, which are only needed in training. Defaults to None.
            test_mode (bool): Whether in test mode. Defaults to False.

        Returns:
                torch.Tensor: The extracted features.
                dict: A dict recording the kwargs for downstream
                    pipeline. These keys are usually included:
                    ``loss_aux``.
        """
        # load checkpoints for edge model and cloud model
        if not self.ckpts_loaded:
            self.load_stage1_ckpts()

        # Record the kwargs required by `loss` and `predict`
        loss_predict_kwargs = dict()

        self.edge_inputs = inputs.clone()

        num_segs = inputs.shape[1]
        # [N, num_crops, C, T, H, W] ->
        # [N * num_crops, C, T, H, W]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        inputs = inputs.view((-1, ) + inputs.shape[2:])

        # Check settings of test
        if test_mode:
            if self.test_cfg is not None:
                loss_predict_kwargs['fcn_test'] = self.test_cfg.get(
                    'fcn_test', False)
            if self.test_cfg is not None and self.test_cfg.get(
                    'max_testing_views', False):
                max_testing_views = self.test_cfg.get('max_testing_views')
                assert isinstance(max_testing_views, int)

                total_views = inputs.shape[0]
                assert num_segs == total_views, (
                    'max_testing_views is only compatible '
                    'with batch_size == 1')
                view_ptr = 0
                feats = []
                while view_ptr < total_views:
                    batch_imgs = inputs[view_ptr:view_ptr + max_testing_views]
                    feat = self.backbone(batch_imgs)
                    if self.with_neck:
                        feat, _ = self.neck(feat)
                    feats.append(feat)
                    view_ptr += max_testing_views

                def recursively_cat(feats):
                    # recursively traverse feats until it's a tensor,
                    # then concat
                    out_feats = []
                    for e_idx, elem in enumerate(feats[0]):
                        batch_elem = [feat[e_idx] for feat in feats]
                        if not isinstance(elem, torch.Tensor):
                            batch_elem = recursively_cat(batch_elem)
                        else:
                            batch_elem = torch.cat(batch_elem)
                        out_feats.append(batch_elem)

                    return tuple(out_feats)

                if isinstance(feats[0], tuple):
                    x = recursively_cat(feats)
                else:
                    x = torch.cat(feats)
            else:
                [edge_output, edge_intermediate_feats], _ = self.edge_model.extract_feat(self.edge_inputs, stage='backbone')
                delivered_edge_feat = edge_intermediate_feats[self.edge_feat_idx]
                delivered_edge_feat = self.edge_model.neck(delivered_edge_feat.clone())
                x = self.backbone(delivered_edge_feat)
                if self.with_neck:
                    x, _ = self.neck(x)

            return x, loss_predict_kwargs
        else:
            # Return features extracted through backbone

            # get edge feats
            [edge_output, edge_intermediate_feats], _ = self.edge_model.extract_feat(self.edge_inputs, stage='backbone')
            delivered_edge_feat = edge_intermediate_feats[self.edge_feat_idx] # [8, 32, 28, 28]
            
            # save edge output
            self.edge_output = [edge_output[0].clone(), edge_output[1].clone()]

            # DFC module: compress & reconstruction
            delivered_edge_feat = self.edge_model.neck(delivered_edge_feat.clone()) # [8, 32, 28, 28]

            # cloud model forward
            x = self.backbone(delivered_edge_feat)
            if stage == 'backbone':
                return x, loss_predict_kwargs
            
            # save preds for self distillation
            self.cloud_preds = self.cls_head(x).clone()
            self.edge_preds = self.edge_model.cls_head(edge_output).clone()

            loss_aux = dict()
            if self.with_neck:
                x, loss_aux = self.neck(x, data_samples=data_samples)

            # Return features extracted through neck
            loss_predict_kwargs['loss_aux'] = loss_aux
            if stage == 'neck':
                return x, loss_predict_kwargs

            # Return raw logits through head.
            if self.with_cls_head and stage == 'head':
                x = self.cls_head(x, **loss_predict_kwargs)
                return x, loss_predict_kwargs

    def loss(self, inputs, data_samples, **kwargs):

        # compute cloud model loss
        loss = super().loss(inputs, data_samples, **kwargs)

        # compute edge model loss
        edge_cls_loss = self.edge_model.cls_head.loss(self.edge_output, data_samples)
        
        # compute self distillation loss
        loss_self_distill = self.distill_loss(self.cloud_preds, self.edge_preds)

        # update loss dict
        loss.update(
            dict(
                edge_top1_acc=edge_cls_loss['top1_acc'],
                edge_top5_acc=edge_cls_loss['top5_acc'],
                edge_loss=edge_cls_loss['loss_cls'],
                loss_self_distill=loss_self_distill,
            )
        )

        return loss