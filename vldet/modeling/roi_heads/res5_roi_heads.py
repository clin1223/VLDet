# Copyright (c) Facebook, Inc. and its affiliates.
import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads
from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads, _ScaleGradient
from detectron2.modeling.roi_heads.box_head import build_box_head

from .vldet_fast_rcnn import VLDetFastRCNNOutputLayers
from ..debug import debug_second_stage

from torch.cuda.amp import autocast

@ROI_HEADS_REGISTRY.register()
class CustomRes5ROIHeads(Res5ROIHeads):
    @configurable
    def __init__(self, **kwargs):
        cfg = kwargs.pop('cfg')
        super().__init__(**kwargs)
        stage_channel_factor = 2 ** 3
        out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor

        self.with_image_labels = cfg.WITH_IMAGE_LABELS
        self.ws_num_props = cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS
        self.add_image_box = cfg.MODEL.ROI_BOX_HEAD.ADD_IMAGE_BOX
        self.add_feature_to_prop = cfg.MODEL.ROI_BOX_HEAD.ADD_FEATURE_TO_PROP
        self.image_box_size = cfg.MODEL.ROI_BOX_HEAD.IMAGE_BOX_SIZE
        self.dis_loss_weight = cfg.MODEL.ROI_BOX_HEAD.DIS_LOSS_WEIGHT
        self.box_predictor = VLDetFastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

        self.save_debug = cfg.SAVE_DEBUG
        self.save_debug_path = cfg.SAVE_DEBUG_PATH
        if self.save_debug:
            self.debug_show_name = cfg.DEBUG_SHOW_NAME
            self.vis_thresh = cfg.VIS_THRESH
            self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(
                torch.device(cfg.MODEL.DEVICE)).view(3, 1, 1)
            self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(
                torch.device(cfg.MODEL.DEVICE)).view(3, 1, 1)
            self.bgr = (cfg.INPUT.FORMAT == 'BGR')

        self.use_caption = cfg.MODEL.ROI_BOX_HEAD.USE_CAPTION
        self.use_ot = cfg.MODEL.ROI_BOX_HEAD.USE_OT
        self.use_distill = cfg.MODEL.ROI_BOX_HEAD.USE_DISTILL

        self.disloss = nn.KLDivLoss(reduction='batchmean')

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['cfg'] = cfg
        return ret

    def _shared_roi_transform(self, features: List[torch.Tensor], boxes: List[Boxes]):
        x = self.pooler(features, boxes)
        return self.res5(x)



    def forward(self, images, features, proposals, targets=None,
        ann_type='box', classifier_info=(None,None,None)):
        '''
        enable debug and image labels
        classifier_info is shared across the batch
        '''
        if not self.save_debug:
            del images
        
        if self.training:
            if ann_type in ['box']:
                proposals = self.label_and_sample_proposals(
                    proposals, targets)
            else:
                proposals = self.get_top_proposals(proposals)

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
            )

        
        predictions = self.box_predictor(
            box_features.mean(dim=[2, 3]), ann_type,
            classifier_info=classifier_info)
        
        if self.add_feature_to_prop:
            feats_per_image = box_features.mean(dim=[2, 3]).split(
                [len(p) for p in proposals], dim=0)
            for feat, p in zip(feats_per_image, proposals):
                p.feat = feat

        if self.training:
            del features
            if (ann_type != 'box'):
                image_labels = [x._pos_category_ids for x in targets]

                losses = {}
                if self.use_caption:
                    caption_losses = self.box_predictor.image_label_losses(
                        predictions, proposals, image_labels,
                        classifier_info=classifier_info,
                        ann_type=ann_type)
                    losses.update({'caption_loss': caption_losses})

                if self.use_ot == 'BCE':
                    ot_loss = self.box_predictor.align_BCE_loss(
                        predictions, proposals, targets,
                        classifier_info=classifier_info,
                        ann_type=ann_type)
                    losses.update({'ot_loss': ot_loss})

                elif self.use_ot == 'contrastive':
                    ot_loss = self.box_predictor.align_contrastive_loss(
                        predictions, proposals, targets,
                        classifier_info=classifier_info,
                        ann_type=ann_type)
                    losses.update({'ot_loss': ot_loss})

                elif self.use_ot == 'sinkhorn':
                    ot_loss = self.box_predictor.align_sinkhorn_loss(
                        predictions, proposals, targets,
                        classifier_info=classifier_info,
                        ann_type=ann_type)
                    losses.update({'ot_loss': ot_loss})

                assert 'loss_cls' not in losses
                assert 'loss_box_reg' not in losses
                losses['loss_cls'] = predictions[0].new_zeros([1])[0]
                losses['loss_box_reg'] = predictions[0].new_zeros([1])[0]
            else:
                losses = self.box_predictor.losses(
                    (predictions[0], predictions[1]), proposals)
                if self.with_image_labels:
                    if self.use_caption:
                        assert 'caption_loss' not in losses
                        losses['caption_loss'] = predictions[0].new_zeros([1])[0]
                    if self.use_ot:
                        assert 'ot_loss' not in losses
                        losses['ot_loss'] = predictions[0].new_zeros([1])[0]
            if self.save_debug:
                denormalizer = lambda x: x * self.pixel_std + self.pixel_mean
                if ann_type != 'box':
                    image_labels = [x._pos_category_ids for x in targets]
                else:
                    image_labels = [[] for x in targets]
                debug_second_stage(
                    [denormalizer(x.clone()) for x in images],
                    targets, proposals=proposals,
                    save_debug=self.save_debug,
                    debug_show_name=self.debug_show_name,
                    vis_thresh=self.vis_thresh,
                    image_labels=image_labels,
                    save_debug_path=self.save_debug_path,
                    bgr=self.bgr)
            return proposals, losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            if self.save_debug:
                denormalizer = lambda x: x * self.pixel_std + self.pixel_mean
                debug_second_stage(
                    [denormalizer(x.clone()) for x in images],
                    pred_instances, proposals=proposals,
                    save_debug=self.save_debug,
                    debug_show_name=self.debug_show_name,
                    vis_thresh=self.vis_thresh,
                    save_debug_path=self.save_debug_path,
                    bgr=self.bgr)
            return pred_instances, {}

    def get_top_proposals(self, proposals):
        for i in range(len(proposals)):
            proposals[i].proposal_boxes.clip(proposals[i].image_size)
        proposals = [p[:self.ws_num_props] for p in proposals]
        for i, p in enumerate(proposals):
            p.proposal_boxes.tensor = p.proposal_boxes.tensor.detach()
            if self.add_image_box:
                proposals[i] = self._add_image_box(p)
        return proposals

    def _add_image_box(self, p, use_score=False):
        image_box = Instances(p.image_size)
        n = 1
        h, w = p.image_size
        if self.image_box_size < 1.0:
            f = self.image_box_size
            image_box.proposal_boxes = Boxes(
                p.proposal_boxes.tensor.new_tensor(
                    [w * (1. - f) / 2., 
                        h * (1. - f) / 2.,
                        w * (1. - (1. - f) / 2.), 
                        h * (1. - (1. - f) / 2.)]
                    ).view(n, 4))
        else:
            image_box.proposal_boxes = Boxes(
                p.proposal_boxes.tensor.new_tensor(
                    [0, 0, w, h]).view(n, 4))
        if use_score:
            image_box.scores = \
                p.objectness_logits.new_ones(n)
            image_box.pred_classes = \
                p.objectness_logits.new_zeros(n, dtype=torch.long) 
            image_box.objectness_logits = \
                p.objectness_logits.new_ones(n) 
        else:
            image_box.objectness_logits = \
                p.objectness_logits.new_ones(n)
        return Instances.cat([p, image_box])

