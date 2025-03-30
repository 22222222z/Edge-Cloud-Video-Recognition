# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union

from mmengine.model import merge_dict
from mmengine.runner.checkpoint import load_checkpoint
from mmaction.registry import MODELS
from mmaction.utils import SampleList
from mmaction.evaluation import top_k_accuracy
from .base import BaseRecognizer

from mmengine.logging import MMLogger
logger = MMLogger.get_current_instance()

@MODELS.register_module()
class Recognizer2DDistill(BaseRecognizer):
    """2D recognizer model framework."""
    def __init__(self, backbone, teacher_ckpt, teacher_model, distill_loss, cls_head = None, neck = None, train_cfg = None, test_cfg = None, data_preprocessor = None, topk: Union[int, Tuple[int]] = (1, 5)):
    # def __init__(self, backbone, teacher_model, distill_loss, cls_head = None, neck = None, train_cfg = None, test_cfg = None, data_preprocessor = None, topk: Union[int, Tuple[int]] = (1, 5)):
        super().__init__(backbone, cls_head, neck, train_cfg, test_cfg, data_preprocessor)

        assert teacher_model is not None, "No teacher model for class Recognizer2DDistill"
        assert distill_loss is not None, "No distillation loss"

        # init teacher model
        self.teacher_model = MODELS.build(teacher_model)
        self.teacher_ckpt, self.teacher_loaded = teacher_ckpt, False

        # for teacher metrics
        self.num_classes = cls_head['num_classes'] if cls_head is not None else None
        self.topk = topk

        # distillation loss
        self.distill_loss = MODELS.build(distill_loss)

    def extract_feat(self,
                     inputs: torch.Tensor,
                     stage: str = 'neck',
                     data_samples: SampleList = None,
                     test_mode: bool = False) -> tuple:
        """Extract features of different stages.

        Args:
            inputs (Tensor): The input data.
            stage (str): Which stage to output the feature.
                Defaults to ``neck``.
            data_samples (List[:obj:`ActionDataSample`]): Action data
                samples, which are only needed in training. Defaults to None.
            test_mode: (bool): Whether in test mode. Defaults to False.

        Returns:
                Tensor: The extracted features.
                dict: A dict recording the kwargs for downstream
                    pipeline. These keys are usually included:
                    ``num_segs``, ``fcn_test``, ``loss_aux``.
        """

        # Record the kwargs required by `loss` and `predict`.
        loss_predict_kwargs = dict()

        teacher_inputs = None
        if len(inputs.size()) == 6:
            teacher_inputs = inputs.clone()
            inputs = inputs.squeeze(1).permute(0, 2, 1, 3, 4).contiguous()
        else:
            pass

        num_segs = inputs.shape[1]
        loss_predict_kwargs['num_segs'] = num_segs

        # [N, num_crops * num_segs, C, H, W] -> [N, num_crops, C, T, H, W]
        if teacher_inputs == None:
            N, _, C, H, W = inputs.shape
            teacher_inputs = inputs.clone().reshape(N, -1, C, num_segs, H, W)

        # [N, num_crops * num_segs, C, H, W] ->
        # [N * num_crops * num_segs, C, H, W]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        inputs = inputs.view((-1, ) + inputs.shape[2:])

        def forward_once(batch_imgs):
            # Extract features through backbone.
            if (hasattr(self.backbone, 'features')
                    and self.backbone_from == 'torchvision'):
                x = self.backbone.features(batch_imgs)
            elif self.backbone_from == 'timm':
                x = self.backbone.forward_features(batch_imgs)
            elif self.backbone_from in ['mmcls', 'mmpretrain']:
                x = self.backbone(batch_imgs)
                if isinstance(x, tuple):
                    assert len(x) == 1
                    x = x[0]
            else:
                x = self.backbone(batch_imgs)

            if self.backbone_from in ['torchvision', 'timm']:
                if not self.feature_shape:
                    # Transformer-based feature shape: B x L x C.
                    if len(x.shape) == 3:
                        self.feature_shape = 'NLC'
                    # Resnet-based feature shape: B x C x Hs x Ws.
                    elif len(x.shape) == 4:
                        self.feature_shape = 'NCHW'

                if self.feature_shape == 'NHWC':
                    x = nn.AdaptiveAvgPool2d(1)(x.permute(0, 3, 1,
                                                          2))  # B x C x 1 x 1
                elif self.feature_shape == 'NCHW':
                    x = nn.AdaptiveAvgPool2d(1)(x)  # B x C x 1 x 1
                elif self.feature_shape == 'NLC':
                    x = nn.AdaptiveAvgPool1d(1)(x.transpose(1, 2))  # B x C x 1

                x = x.reshape((x.shape[0], -1))  # B x C
                x = x.reshape(x.shape + (1, 1))  # B x C x 1 x 1
            return x

        # Check settings of `fcn_test`.
        fcn_test = False
        if test_mode:
            if self.test_cfg is not None and self.test_cfg.get(
                    'fcn_test', False):
                fcn_test = True
                num_segs = self.test_cfg.get('num_segs',
                                             self.backbone.num_segments)
            loss_predict_kwargs['fcn_test'] = fcn_test

            # inference with batch size of `max_testing_views` if set
            if self.test_cfg is not None and self.test_cfg.get(
                    'max_testing_views', False):
                max_testing_views = self.test_cfg.get('max_testing_views')
                assert isinstance(max_testing_views, int)
                # backbone specify num_segments
                num_segments = self.backbone.get('num_segments')
                if num_segments is not None:
                    assert max_testing_views % num_segments == 0, \
                        'make sure that max_testing_views is a multiple of ' \
                        'num_segments, but got {max_testing_views} and '\
                        '{num_segments}'

                total_views = inputs.shape[0]
                view_ptr = 0
                feats = []
                while view_ptr < total_views:
                    batch_imgs = inputs[view_ptr:view_ptr + max_testing_views]
                    feat = forward_once(batch_imgs)
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
                x = forward_once(inputs)
        else:
            x = forward_once(inputs)

        # Return features extracted through backbone.
        if stage == 'backbone':
            return x, loss_predict_kwargs
        
        # save results for distillation loss
        self.preds = self.cls_head(x).clone()
        self.teacher_inputs = teacher_inputs

        loss_aux = dict()
        if self.with_neck:
            # x is a tuple with multiple feature maps.
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, loss_aux = self.neck(x, data_samples=data_samples)
            if not fcn_test:
                x = x.squeeze(2)
                loss_predict_kwargs['num_segs'] = 1
        elif fcn_test:
            # full convolution (fcn) testing when no neck
            # [N * num_crops * num_segs, C', H', W'] ->
            # [N * num_crops, C', num_segs, H', W']
            x = x.reshape((-1, num_segs) +
                          x.shape[1:]).transpose(1, 2).contiguous()

        loss_predict_kwargs['loss_aux'] = loss_aux

        # Return features extracted through neck.
        if stage == 'neck':
            return x, loss_predict_kwargs

        # Return raw logits through head.
        if self.with_cls_head and stage == 'head':
            # [N * num_crops, num_classes]
            x = self.cls_head(x, **loss_predict_kwargs)
            return x, loss_predict_kwargs
        
    def get_distill_metrics(self, preds, teacher_inputs, data_samples, **kwargs):
        
        # init loss for distilation
        losses = dict()

        # get teacher preds
        with torch.no_grad():
            self.teacher_model.eval()
            cls_scores, _ = self.teacher_model.extract_feat(teacher_inputs, stage='head') # , test_mode=True)

        # compute distillation loss
        losses["loss_distill"] = self.distill_loss(preds, cls_scores)

        # compute teacher metrics
        labels = [x.gt_label for x in data_samples]
        labels = torch.stack(labels).to(cls_scores.device)
        labels = labels.squeeze()

        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_scores.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_score` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if cls_scores.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_scores.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'teacher_top{k}_acc'] = torch.tensor(
                    a, device=cls_scores.device)

        return losses
        
    def loss(self, inputs, data_samples, **kwargs):

        # cross entropy loss
        loss = super().loss(inputs, data_samples, **kwargs)

        # load pretrained teacher ckpt
        if not self.teacher_loaded:
            logger.info(f'Loading pretrained teacher checkpoint form {self.teacher_ckpt}')
            load_checkpoint(
                self.teacher_model,
                self.teacher_ckpt,
                map_location="cpu",
                strict=False,
            )
            self.teacher_loaded = True

        # distillation loss
        distill_loss = self.get_distill_metrics(self.preds, self.teacher_inputs, data_samples)
        self.preds, self.teacher_inputs = None, None

        # update loss dict
        loss = merge_dict(loss, distill_loss)

        return loss

