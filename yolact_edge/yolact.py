import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import product
from math import sqrt
from typing import List
from config import cfg, mask_type
from yolact_edge.backbone import construct_backbone
import logging


use_torch2trt = False
device = 'cpu'
use_jit = False if use_torch2trt else torch.cuda.device_count() <= 1
ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module

def conv_lrelu(in_features, out_features, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(
            in_features,
            out_features,
            kernel_size=kernel_size,
            stride=stride, padding=dilation,
            dilation=dilation, groups=groups,
        ),
        nn.LeakyReLU(0.1, inplace=True)
    )

class Concat(nn.Module):
    def __init__(self, nets, extra_params):
        super().__init__()

        self.nets = nn.ModuleList(nets)
        self.extra_params = extra_params

    def forward(self, x):
        # Concat each along the channel dimension
        return torch.cat([net(x) for net in self.nets], dim=1, **self.extra_params)

class InterpolateModule(nn.Module):
    """
    This is a module version of F.interpolate (rip nn.Upsampling).
    Any arguments you give it just get passed along for the ride.
    """

    def __init__(self, *args, **kwdargs):
        super().__init__()

        self.args = args
        self.kwdargs = kwdargs

    def forward(self, x):
        return F.interpolate(x, *self.args, **self.kwdargs)


def make_net(in_channels, conf, include_last_relu=True):
    """
    A helper function to take a config setting and turn it into a network.
    Used by protonet and extrahead. Returns (network, out_channels)
    """
    def make_layer(layer_cfg):
        nonlocal in_channels

        # Possible patterns:
        # ( 256, 3, {}) -> conv
        # ( 256,-2, {}) -> deconv
        # (None,-2, {}) -> bilinear interpolate
        # ('cat',[],{}) -> concat the subnetworks in the list
        #
        # You know it would have probably been simpler just to adopt a 'c' 'd' 'u' naming scheme.
        # Whatever, it's too late now.
        if isinstance(layer_cfg[0], str):
            layer_name = layer_cfg[0]

            if layer_name == 'cat':
                nets = [make_net(in_channels, x) for x in layer_cfg[1]]
                layer = Concat([net[0] for net in nets], layer_cfg[2])
                num_channels = sum([net[1] for net in nets])
        else:
            num_channels = layer_cfg[0]
            kernel_size = layer_cfg[1]

            if kernel_size > 0:
                layer = nn.Conv2d(in_channels, num_channels, kernel_size, **layer_cfg[2])
            else:
                if num_channels is None:
                    layer = InterpolateModule(scale_factor=-kernel_size, mode='bilinear', align_corners=False, **layer_cfg[2])
                else:
                    layer = nn.ConvTranspose2d(in_channels, num_channels, -kernel_size, **layer_cfg[2])

        in_channels = num_channels if num_channels is not None else in_channels

        # Don't return a ReLU layer if we're doing an upsample. This probably doesn't affect anything
        # output-wise, but there's no need to go through a ReLU here.
        # Commented out for backwards compatibility with previous models
        # if num_channels is None:
        #     return [layer]
        # else:
        return [layer, nn.ReLU(inplace=True)]

    # Use sum to concat together all the component layer lists
    net = sum([make_layer(x) for x in conf], [])
    if not include_last_relu:
        net = net[:-1]

    return nn.Sequential(*(net)), in_channels

class SPA(ScriptModuleWrapper):
    __constants__ = ['interpolation_mode', 'use_normalized_spa']

    def __init__(self, num_layers):
        super().__init__()
        self.interpolation_mode = cfg.fpn.interpolation_mode
        self.use_normalized_spa = cfg.flow.use_normalized_spa

        self.refine_convs = nn.ModuleList([
            conv_lrelu(cfg.fpn.num_features * 2, cfg.fpn.num_features)
            for _ in range(num_layers - 1)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, 0.1, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, c3, f2, f3):
        fpn_outs = [f2, f3]
        out = []

        j = 0

        for refine in self.refine_convs:
            x = fpn_outs[j]
            _, _, h, w = x.size()
            c3 = F.interpolate(c3, size=(h, w), mode=self.interpolation_mode, align_corners=False)
            x_normalize = F.normalize(x, dim=1)
            c3_normalize = F.normalize(c3, dim=1)
            if self.use_normalized_spa:
                x = x + refine(torch.cat((x_normalize, c3_normalize), dim=1))
            else:
                x = x + refine(torch.cat((x, c3), dim=1))
            out.append(x)
            j += 1
        return out

class FPN_phase_1(ScriptModuleWrapper):
    __constants__ = ['interpolation_mode']

    def __init__(self, in_channels):
        super().__init__()

        self.src_channels = in_channels
        self.lat_layers = nn.ModuleList([
            nn.Conv2d(x, cfg.fpn.num_features, kernel_size=1)
            for x in reversed(in_channels)
        ])
        self.interpolation_mode = cfg.fpn.interpolation_mode


    def forward(self, x1=None, x2=None, x3=None, x4=None, x5=None, x6=None, x7=None):
        """
        Args:
            - convouts (list): A list of convouts for the corresponding layers in in_channels.
        Returns:
            - A list of FPN convouts in the same order as x with extra downsample layers if requested.
        """
        convouts_ = [x1, x2, x3, x4, x5, x6, x7]
        convouts = []
        j = 0
        while j < len(convouts_):
            t = convouts_[j]
            if t is not None:
                convouts.append(t)
            j += 1

        out = []
        lat_feats = []
        x = torch.zeros(1, device=convouts[0].device)
        for i in range(len(convouts)):
            out.append(x)
            lat_feats.append(x)

        # For backward compatability, the conv layers are stored in reverse but the input and output is
        # given in the correct order. Thus, use j=-i-1 for the input and output and i for the conv layers.
        j = len(convouts)
        for lat_layer in self.lat_layers:
            j -= 1

            if j < len(convouts) - 1:
                _, _, h, w = convouts[j].size()
                x = F.interpolate(x, size=(h, w), mode=self.interpolation_mode, align_corners=False)
            lat_j = lat_layer(convouts[j])
            lat_feats[j] = lat_j
            x = x + lat_j
            out[j] = x

        for i in range(len(convouts)):
            out.append(lat_feats[i])
        return out


class FPN_phase_2(ScriptModuleWrapper):
    __constants__ = ['num_downsample', 'use_conv_downsample']

    def __init__(self, in_channels):
        super().__init__()

        self.src_channels = in_channels

        # This is here for backwards compatability
        padding = 1 if cfg.fpn.pad else 0
        self.pred_layers = nn.ModuleList([
            nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=padding)
            for _ in in_channels
        ])

        if cfg.fpn.use_conv_downsample:
            self.downsample_layers = nn.ModuleList([
                nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=1, stride=2)
                for _ in range(cfg.fpn.num_downsample)
            ])

        self.num_downsample = cfg.fpn.num_downsample
        self.use_conv_downsample = cfg.fpn.use_conv_downsample

    def forward(self, x1=None, x2=None, x3=None, x4=None, x5=None, x6=None, x7=None):
        """
        Args:
            - convouts (list): A list of convouts for the corresponding layers in in_channels.
        Returns:
            - A list of FPN convouts in the same order as x with extra downsample layers if requested.
        """
        out_ = [x1, x2, x3, x4, x5, x6, x7]
        out = []
        j = 0
        while j < len(out_):
            t = out_[j]
            if t is not None:
                out.append(t)
            j += 1

        len_convouts = len(out)

        j = len_convouts
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = F.relu(pred_layer(out[j]))

        # In the original paper, this takes care of P6
        if self.use_conv_downsample:
            for downsample_layer in self.downsample_layers:
                out.append(downsample_layer(out[-1]))
        else:
            for idx in range(self.num_downsample):
                # Note: this is an untested alternative to out.append(out[-1][:, :, ::2, ::2]). Thanks TorchScript.
                out.append(nn.functional.max_pool2d(out[-1], 1, stride=2))

        return out

class FPN(ScriptModuleWrapper):
    """
    Implements a general version of the FPN introduced in
    https://arxiv.org/pdf/1612.03144.pdf

    Parameters (in cfg.fpn):
        - num_features (int): The number of output features in the fpn layers.
        - interpolation_mode (str): The mode to pass to F.interpolate.
        - num_downsample (int): The number of downsampled layers to add onto the selected layers.
                                These extra layers are downsampled from the last selected layer.

    Args:
        - in_channels (list):
            For each conv layer you supply in the forward pass,
            how many features will it have?
    """
    __constants__ = ['interpolation_mode', 'num_downsample', 'use_conv_downsample']

    def __init__(self, in_channels):
        super().__init__()

        self.lat_layers  = nn.ModuleList([
            nn.Conv2d(x, cfg.fpn.num_features, kernel_size=1)
            for x in reversed(in_channels)
        ])

        # This is here for backwards compatability
        padding = 1 if cfg.fpn.pad else 0
        self.pred_layers = nn.ModuleList([
            nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=padding)
            for _ in in_channels
        ])

        if cfg.fpn.use_conv_downsample:
            self.downsample_layers = nn.ModuleList([
                nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=1, stride=2)
                for _ in range(cfg.fpn.num_downsample)
            ])

        self.interpolation_mode  = cfg.fpn.interpolation_mode
        self.num_downsample      = cfg.fpn.num_downsample
        self.use_conv_downsample = cfg.fpn.use_conv_downsample

    def forward(self, convouts:List[torch.Tensor]):
        """
        Args:
            - convouts (list): A list of convouts for the corresponding layers in in_channels.
        Returns:
            - A list of FPN convouts in the same order as x with extra downsample layers if requested.
        """

        out = []
        x = torch.zeros(1, device=convouts[0].device)
        for i in range(len(convouts)):
            out.append(x)

        # For backward compatability, the conv layers are stored in reverse but the input and output is
        # given in the correct order. Thus, use j=-i-1 for the input and output and i for the conv layers.
        j = len(convouts)
        for lat_layer in self.lat_layers:
            j -= 1

            if j < len(convouts) - 1:
                _, _, h, w = convouts[j].size()
                x = F.interpolate(x, size=(h, w), mode=self.interpolation_mode, align_corners=False)
            x = x + lat_layer(convouts[j])
            out[j] = x

        # This janky second loop is here because TorchScript.
        j = len(convouts)
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = F.relu(pred_layer(out[j]))

        # In the original paper, this takes care of P6
        if self.use_conv_downsample:
            for downsample_layer in self.downsample_layers:
                out.append(downsample_layer(out[-1]))
        else:
            for idx in range(self.num_downsample):
                # Note: this is an untested alternative to out.append(out[-1][:, :, ::2, ::2]). Thanks TorchScript.
                out.append(nn.functional.max_pool2d(out[-1], 1, stride=2))

        return out

class PredictionModule(nn.Module):
    """
    The (c) prediction module adapted from DSSD:
    https://arxiv.org/pdf/1701.06659.pdf

    Note that this is slightly different to the module in the paper
    because the Bottleneck block actually has a 3x3 convolution in
    the middle instead of a 1x1 convolution. Though, I really can't
    be arsed to implement it myself, and, who knows, this might be
    better.

    Args:
        - in_channels:   The input feature size.
        - out_channels:  The output feature size (must be a multiple of 4).
        - aspect_ratios: A list of lists of priorbox aspect ratios (one list per scale).
        - scales:        A list of priorbox scales relative to this layer's convsize.
                        For instance: If this layer has convouts of size 30x30 for
                                    an image of size 600x600, the 'default' (scale
                                    of 1) for this layer would produce bounding
                                    boxes with an area of 20x20px. If the scale is
                                    .5 on the other hand, this layer would consider
                                    bounding boxes with area 10x10px, etc.
        - parent:       If parent is a PredictionModule, this module will use all the layers
                        from parent instead of from this module.
    """

    def __init__(self, in_channels, out_channels=1024, aspect_ratios=[[1]], scales=[1], parent=None, index=0):
        super().__init__()

        self.params = [in_channels, out_channels, aspect_ratios, scales, parent, index]

        self.num_classes = cfg.num_classes
        self.mask_dim    = cfg.mask_dim
        self.num_priors  = sum(len(x) for x in aspect_ratios)
        self.parent      = [parent] # Don't include this in the state dict
        self.index       = index

        if cfg.mask_proto_prototypes_as_features:
            in_channels += self.mask_dim

        if parent is None:
            if cfg.extra_head_net is None:
                out_channels = in_channels
            else:
                self.upfeature, out_channels = make_net(in_channels, cfg.extra_head_net)

            self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4,                **cfg.head_layer_params)
            self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.num_classes, **cfg.head_layer_params)
            self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim,    **cfg.head_layer_params)

            if cfg.use_instance_coeff:
                self.inst_layer = nn.Conv2d(out_channels, self.num_priors * cfg.num_instance_coeffs, **cfg.head_layer_params)

            # What is this ugly lambda doing in the middle of all this clean prediction module code?
            def make_extra(num_layers):
                if num_layers == 0:
                    return lambda x: x
                else:
                    # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                    return nn.Sequential(*sum([[
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    ] for _ in range(num_layers)], []))

            self.bbox_extra, self.conf_extra, self.mask_extra = [make_extra(x) for x in cfg.extra_layers]

            if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_coeff_gate:
                self.gate_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, kernel_size=3, padding=1)

        self.aspect_ratios = aspect_ratios
        self.scales = scales

        self.priors = None
        self.last_conv_size = None

        # For use in evaluation
        self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=100, conf_thresh=0.05, nms_thresh=0.5)


    def forward(self, x):
        """
        Args:
            - x: The convOut from a layer in the backbone network
                Size: [batch_size, in_channels, conv_h, conv_w])

        Returns a tuple (bbox_coords, class_confs, mask_output, prior_boxes) with sizes
            - bbox_coords: [batch_size, conv_h*conv_w*num_priors, 4]
            - class_confs: [batch_size, conv_h*conv_w*num_priors, num_classes]
            - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_dim]
            - prior_boxes: [conv_h*conv_w*num_priors, 4]
        """
        # In case we want to use another module's layers
        src = self if self.parent[0] is None else self.parent[0]

        conv_h = x.size(2)
        conv_w = x.size(3)

        if cfg.extra_head_net is not None:
            x = src.upfeature(x)

        if cfg.use_prediction_module:
            # The two branches of PM design (c)
            a = src.block(x)

            b = src.conv(x)
            b = src.bn(b)
            b = F.relu(b)

            # TODO: Possibly switch this out for a product
            x = a + b

        bbox_x = src.bbox_extra(x)
        conf_x = src.conf_extra(x)
        mask_x = src.mask_extra(x)

        bbox = src.bbox_layer(bbox_x).permute(0, 2, 3, 1).contiguous()
        bbox = bbox.view(bbox.size(0), (bbox.size()[0]*bbox.size()[1]*bbox.size()[2]*bbox.size()[3])//4, 4)
        conf = src.conf_layer(conf_x).permute(0, 2, 3, 1).contiguous()
        conf = conf.view(conf.size(0), (conf.size()[0]*conf.size()[1]*conf.size()[2]*conf.size()[3])//self.num_classes, self.num_classes)

        if cfg.eval_mask_branch:
            mask = src.mask_layer(mask_x).permute(0, 2, 3, 1).contiguous()
            mask = mask.view(mask.size(0), (mask.size()[0]*mask.size()[1]*mask.size()[2]*mask.size()[3])//self.mask_dim, self.mask_dim)
        else:
            mask = torch.zeros(x.size(0), bbox.size(1), self.mask_dim, device=bbox.device)

        if cfg.use_instance_coeff:
            inst = src.inst_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, cfg.num_instance_coeffs)

        # See box_utils.decode for an explanation of this
        if cfg.use_yolo_regressors:
            bbox[:, :, :2] = torch.sigmoid(bbox[:, :, :2]) - 0.5
            bbox[:, :, 0] /= conv_w
            bbox[:, :, 1] /= conv_h

        if cfg.eval_mask_branch:
            if cfg.mask_type == mask_type.direct:
                mask = torch.sigmoid(mask)
            elif cfg.mask_type == mask_type.lincomb:
                mask = cfg.mask_proto_coeff_activation(mask)

                if cfg.mask_proto_coeff_gate:
                    gate = src.gate_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
                    mask = mask * torch.sigmoid(gate)

        # priors = self.make_priors(conv_h, conv_w)
        self.make_priors(conv_h, conv_w)

        preds = { 'loc': bbox, 'conf': conf, 'mask': mask, 'priors': self.priors }

        if cfg.use_instance_coeff:
            preds['inst'] = inst

        return preds

    def make_priors(self, conv_h, conv_w):
        if self.last_conv_size != (conv_w, conv_h):
            prior_data = []

            # Iteration order is important (it has to sync up with the convout)
            for j, i in product(range(conv_h), range(conv_w)):
                # +0.5 because priors are in center-size notation
                x = (i + 0.5) / conv_w
                y = (j + 0.5) / conv_h

                for scale, ars in zip(self.scales, self.aspect_ratios):
                    for ar in ars:
                        if not cfg.backbone.preapply_sqrt:
                            ar = sqrt(ar)

                        if cfg.backbone.use_pixel_scales:
                            if type(cfg.max_size) == tuple:
                                width, height = cfg.max_size
                                w = scale * ar / width
                                h = scale / ar / height
                            else:
                                w = scale * ar / cfg.max_size
                                h = scale / ar / cfg.max_size
                        else:
                            w = scale * ar / conv_w
                            h = scale / ar / conv_h

                        # This is for backward compatability with a bug where I made everything square by accident
                        if cfg.backbone.use_square_anchors:
                            h = w

                        prior_data += [x, y, w, h]

            self.priors = torch.Tensor(prior_data).reshape(-1, 4)#.view(-1, 4)
            self.last_conv_size = (conv_w, conv_h)

    def create_partial_backbone(self):
        if cfg.flow.warp_mode == 'none':
            return

        logger = logging.getLogger("yolact.model.load")
        logger.debug("Creating partial backbone...")

        backbone = construct_backbone(cfg.backbone)
        backbone.load_state_dict(self.backbone.state_dict())
        backbone.layers = backbone.layers[:2]

        self.partial_backbone = backbone
        logger.debug("Partial backbone created...")



class Detect(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh

        self.use_cross_class_nms = False
        self.use_fast_nms = False

    def __call__(self, loc, conf, mask, priors):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
            mask_data: (tensor) Mask preds from mask layers
                Shape: [batch, num_priors, mask_dim]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors, 4]
            proto_data: (tensor) If using mask_type.lincomb, the prototype masks
                Shape: [batch, mask_h, mask_w, mask_dim]

        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
        """
        loc_data   = loc
        conf_data  = conf
        mask_data  = mask
        prior_data = priors

        batch_size = loc_data.size(0)
        num_priors = prior_data.size(0)
        conf_preds = conf_data.view(batch_size, num_priors, self.num_classes).transpose(2, 1).contiguous()

        decoded_boxes = self.decode(loc_data, prior_data)
        boxes = decoded_boxes
        conf_preds = conf_preds[:, 1:, :]
        scores, classes = torch.max(conf_preds, dim=1)
        masks = mask_data

        return (boxes, scores, classes, masks)


    def point_form(self, boxes):
        """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
        representation for comparison to point form ground truth data.
        Args:
            boxes: (tensor) center-size default boxes from priorbox layers.
        Return:
            boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
        """
        return torch.cat(
            (
                boxes[:, :, :2] - boxes[:, :, 2:]/2, # xmin, ymin
                boxes[:, :, :2] + boxes[:, :, 2:]/2, # xmax, ymax
            ),
            1,
        )


    def decode(self, loc, priors, use_yolo_regressors:bool=False):
        """
        Decode predicted bbox coordinates using the same scheme
        employed by Yolov2: https://arxiv.org/pdf/1612.08242.pdf

            b_x = (sigmoid(pred_x) - .5) / conv_w + prior_x
            b_y = (sigmoid(pred_y) - .5) / conv_h + prior_y
            b_w = prior_w * exp(loc_w)
            b_h = prior_h * exp(loc_h)

        Note that loc is inputed as [(s(x)-.5)/conv_w, (s(y)-.5)/conv_h, w, h]
        while priors are inputed as [x, y, w, h] where each coordinate
        is relative to size of the image (even sigmoid(x)). We do this
        in the network by dividing by the 'cell size', which is just
        the size of the convouts.

        Also note that prior_x and prior_y are center coordinates which
        is why we have to subtract .5 from sigmoid(pred_x and pred_y).

        Args:
            - loc:    The predicted bounding boxes of size [num_priors, 4]
            - priors: The priorbox coords with size [num_priors, 4]

        Returns: A tensor of decoded relative coordinates in point form
                form with size [num_priors, 4]
        """
        priors = priors[np.newaxis, ...]
        variances = [0.1, 0.2]

        boxes = torch.cat(
            (
                priors[:, :, :2] + loc[:, :, :2] * variances[0] * priors[:, :, 2:],
                priors[:, :, 2:] * torch.exp(loc[:, :, 2:] * variances[1])
            ),
            2
        )

        boxes0 = boxes[:, :, 0] - boxes[:, :, 2] / 2
        boxes1 = boxes[:, :, 1] - boxes[:, :, 3] / 2
        boxes2 = boxes[:, :, 0] + boxes[:, :, 2] / 2
        boxes3 = boxes[:, :, 1] + boxes[:, :, 3] / 2
        boxes = torch.cat(
            [
                boxes0[...,np.newaxis],
                boxes1[...,np.newaxis],
                boxes2[...,np.newaxis],
                boxes3[...,np.newaxis]
            ], dim=2)

        return boxes



    def detect(self, batch_idx, conf_preds, decoded_boxes, mask_data):
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        cur_scores = conf_preds[batch_idx, 1:, :]
        boxes, masks, classes, scores = self.traditional_nms(
            decoded_boxes,
            mask_data,
            cur_scores,
            self.nms_thresh,
            self.conf_thresh
        )
        return boxes, masks, classes, scores


    def traditional_nms(self, boxes, masks, scores, iou_threshold=0.5, conf_thresh=0.05):
        scores = torch.permute(scores, dims=[1,0])
        class_keep = torch.argmax(scores, dim=1)
        max_score_classes = scores[class_keep,-1]
        masks = masks[0]

        boxes = boxes
        scores = max_score_classes
        classes = class_keep
        masks = masks
        boxes = boxes * cfg.max_size

        return boxes / cfg.max_size, masks, classes, scores


class Yolact(nn.Module):

    def __init__(self, training=True):
        super().__init__()

        self.backbone = construct_backbone(cfg.backbone)

        self.training = training

        # Compute mask_dim here and add it back to the config. Make sure Yolact's constructor is called early!
        if cfg.mask_type == mask_type.direct:
            cfg.mask_dim = cfg.mask_size**2
        elif cfg.mask_type == mask_type.lincomb:
            if cfg.mask_proto_use_grid:
                self.grid = torch.Tensor(np.load(cfg.mask_proto_grid_file))
                self.num_grids = self.grid.size(0)
            else:
                self.num_grids = 0

            self.proto_src = cfg.mask_proto_src

            if self.proto_src is None: in_channels = 3
            elif cfg.fpn is not None: in_channels = cfg.fpn.num_features
            else: in_channels = self.backbone.channels[self.proto_src]
            in_channels += self.num_grids

            # The include_last_relu=false here is because we might want to change it to another function
            self.proto_net, cfg.mask_dim = make_net(in_channels, cfg.mask_proto_net, include_last_relu=False)

            if cfg.mask_proto_bias:
                cfg.mask_dim += 1


        self.selected_layers = cfg.backbone.selected_layers
        src_channels = self.backbone.channels

        if cfg.fpn is not None:
            # Some hacky rewiring to accomodate the FPN
            if cfg.flow is not None:
                self.fpn_phase_1 = FPN_phase_1([src_channels[i] for i in self.selected_layers])
                self.fpn_phase_2 = FPN_phase_2([src_channels[i] for i in self.selected_layers])
                if cfg.flow.use_spa:
                    self.spa = SPA(len(self.selected_layers))
                self.selected_layers = list(range(len(self.selected_layers) + cfg.fpn.num_downsample))
            else:
                self.fpn = FPN([src_channels[i] for i in self.selected_layers])
                self.selected_layers = list(range(len(self.selected_layers) + cfg.fpn.num_downsample))
            src_channels = [cfg.fpn.num_features] * len(self.selected_layers)

        self.prediction_layers = nn.ModuleList()

        for idx, layer_idx in enumerate(self.selected_layers):
            # If we're sharing prediction module weights, have every module's parent be the first one
            parent = None
            if cfg.share_prediction_module and idx > 0:
                parent = self.prediction_layers[0]

            pred = PredictionModule(src_channels[layer_idx], src_channels[layer_idx],
                                    aspect_ratios = cfg.backbone.pred_aspect_ratios[idx],
                                    scales        = cfg.backbone.pred_scales[idx],
                                    parent        = parent,
                                    index         = idx)
            self.prediction_layers.append(pred)

        # Extra parameters for the extra losses
        if cfg.use_class_existence_loss:
            # This comes from the smallest layer selected
            # Also note that cfg.num_classes includes background
            self.class_existence_fc = nn.Linear(src_channels[-1], cfg.num_classes - 1)

        if cfg.use_semantic_segmentation_loss:
            self.semantic_seg_conv = nn.Conv2d(src_channels[0], cfg.num_classes-1, kernel_size=1)

        # # For use in evaluation
        # self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=cfg.nms_top_k,
        #     conf_thresh=cfg.nms_conf_thresh, nms_thresh=cfg.nms_thresh)

        self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=100, conf_thresh=0.05, nms_thresh=0.5)

    def forward(self, x, extras=None):
        extras = extras = {"backbone": "full", "interrupt": False, "moving_statistics": {"aligned_feats": []}}
        outs_wrapper = {}

        mean = torch.tensor(
            [
                [
                    [[123.6800]],
                    [[116.7800]],
                    [[103.9400]],
                ],
            ], dtype=torch.float32
        )

        std = torch.tensor(
            [
                [
                    [[58.4000]],
                    [[57.1200]],
                    [[57.3800]],
                ],
            ],
            dtype=torch.float32
        )

        x = (x - mean) / std

        if cfg.flow is None or extras is None or extras["backbone"] == "full":
            outs = self.backbone(x)

        elif extras is not None and extras["backbone"] == "partial":
            if hasattr(self, 'partial_backbone'):
                outs = self.partial_backbone(x)
            else:
                outs = self.backbone(x, partial=True)
        else:
            raise NotImplementedError

        if cfg.flow is not None:
            assert type(extras) == dict
            if extras["backbone"] == "full":
                outs = [outs[i] for i in cfg.backbone.selected_layers]
                outs_fpn_phase_1_wrapper = self.fpn_phase_1(*outs)
                outs_phase_1, lats_phase_1 = outs_fpn_phase_1_wrapper[:len(outs)], outs_fpn_phase_1_wrapper[len(outs):]
                lateral = lats_phase_1[0].detach()
                moving_statistics = extras["moving_statistics"]

                if extras.get("keep_statistics", False):
                    outs_wrapper["feats"] = [out.detach() for out in outs_phase_1]
                    outs_wrapper["lateral"] = lateral

                outs_wrapper["outs_phase_1"] = [out.detach() for out in outs_phase_1]
            else:
                assert extras["moving_statistics"] is not None
                moving_statistics = extras["moving_statistics"]
                outs_phase_1 = moving_statistics["feats"].copy()

                if cfg.flow.warp_mode != 'take':
                    src_conv = outs[-1].detach()
                    src_lat_layer = self.lat_layer if hasattr(self, 'lat_layer') else self.fpn_phase_1.lat_layers[-1]
                    lateral = src_lat_layer(src_conv).detach()

                if cfg.flow.warp_mode == "flow":
                    flows = self.flow_net(self.flow_net_pre_convs(lateral), self.flow_net_pre_convs(moving_statistics["lateral"]))
                    preds_feat = list()
                    if cfg.flow.flow_layer == 'top':
                        flows = [flows[0] for _ in flows]
                    if cfg.flow.warp_layers == 'P4P5':
                        flows = flows[1:]
                        outs_phase_1 = outs_phase_1[1:]
                    for (flow, scale_factor, scale_bias), feat in zip(flows, outs_phase_1):
                        if cfg.flow.flow_layer == 'top':
                            _, _, h, w = feat.size()
                            _, _, h_, w_ = flow.size()
                            if (h, w) != (h_, w_):
                                flow = F.interpolate(flow, size=(h, w), mode="area")
                                scale_factor = F.interpolate(scale_factor, size=(h, w), mode="area")
                                scale_bias = F.interpolate(scale_bias, size=(h, w), mode="area")
                        pred_feat = deform_op(feat, flow)
                        if cfg.flow.use_scale_factor:
                            pred_feat *= scale_factor
                        if cfg.flow.use_scale_bias:
                            pred_feat += scale_bias
                        preds_feat.append(pred_feat)
                    outs_wrapper["preds_flow"] = [[x.detach() for x in flow] for flow in flows]
                    outs_phase_1 = preds_feat

                if cfg.flow.warp_layers == 'P4P5':
                    _, _, h, w = src_conv.size()
                    src_fpn = outs_phase_1[0]
                    src_fpn = F.interpolate(src_fpn, size=(h, w), mode=cfg.fpn.interpolation_mode, align_corners=False)
                    p3 = src_fpn + lateral

                    outs_phase_1 = [p3] + outs_phase_1

                if cfg.flow.use_spa:
                    fpn_outs = outs_phase_1.copy()
                    outs_phase_1 = [fpn_outs[0]]
                    outs_ = self.spa(lateral, *fpn_outs[1:])
                    outs_phase_1.extend(outs_)

                outs_wrapper["outs_phase_1"] = outs_phase_1.copy()

            outs = self.fpn_phase_2(*outs_phase_1)
            if extras["backbone"] == "partial":
                outs_wrapper["outs_phase_2"] = [out for out in outs]
            else:
                outs_wrapper["outs_phase_2"] = [out.detach() for out in outs]
        elif cfg.fpn is not None:
            # Use backbone.selected_layers because we overwrote self.selected_layers
            outs = [outs[i] for i in cfg.backbone.selected_layers]
            outs = self.fpn(outs)

        if extras is not None and extras.get("interrupt", None):
            return outs_wrapper

        proto_out = None

        if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:
            proto_x = x if self.proto_src is None else outs[self.proto_src]

            if self.num_grids > 0:
                grids = self.grid.repeat(proto_x.size(0), 1, 1, 1)
                proto_x = torch.cat([proto_x, grids], dim=1)

            proto_out = self.proto_net(proto_x)

            proto_out = cfg.mask_proto_prototype_activation(proto_out)

            if cfg.mask_proto_prototypes_as_features:
                    # Clone here because we don't want to permute this, though idk if contiguous makes this unnecessary
                proto_downsampled = proto_out.clone()

                if cfg.mask_proto_prototypes_as_features_no_grad:
                    proto_downsampled = proto_out.detach()

                # Move the features last so the multiplication is easy
            proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

            if cfg.mask_proto_bias:
                bias_shape = [x for x in proto_out.size()]
                bias_shape[-1] = 1
                proto_out = torch.cat([proto_out, torch.ones(*bias_shape)], -1)

        loc, conf, mask, priors = [], [], [], []
        for idx, pred_layer in zip(self.selected_layers, self.prediction_layers):
            pred_x = outs[idx]

            if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_prototypes_as_features:
                    # Scale the prototypes down to the current prediction layer's size and add it as inputs
                proto_downsampled = F.interpolate(proto_downsampled, size=outs[idx].size()[2:], mode='bilinear', align_corners=False)
                pred_x = torch.cat([pred_x, proto_downsampled], dim=1)

                # This is re-enabled during training or non-TRT inference.
            if self.training or not (cfg.torch2trt_prediction_module or cfg.torch2trt_prediction_module_int8):
                    # A hack for the way dataparallel works
                if cfg.share_prediction_module and pred_layer is not self.prediction_layers[0]:
                    pred_layer.parent = [self.prediction_layers[0]]

            p = pred_layer(pred_x)
            loc.append(p['loc'])
            conf.append(p['conf'])
            mask.append(p['mask'])
            priors.append(p['priors'])

        loc = torch.cat(loc, 1)
        conf = torch.cat(conf, 1)
        conf = F.softmax(conf, 2)
        mask = torch.cat(mask, 1)
        priors = torch.cat(priors, 0)
        boxes, scores, classes, masks = self.detect(loc, conf, mask, priors)

        return boxes, scores, classes, masks, proto_out



    def load_weights(self, path, args=None):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path, map_location='cpu')

        # Get all possible weights
        cur_state_dict = self.state_dict()
        if args is not None and args.drop_weights is not None:
            drop_weight_keys = args.drop_weights.split(',')

        transfered_from_yolact = False

        for key in list(state_dict.keys()):
            # For backward compatability, remove these (the new variable is called layers)
            if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
                del state_dict[key]

            # Also for backward compatibility with v1.0 weights, do this check
            if key.startswith('fpn.downsample_layers.'):
                if cfg.fpn is not None and int(key.split('.')[2]) >= cfg.fpn.num_downsample:
                    del state_dict[key]

            if args is not None and args.drop_weights is not None:
                for drop_key in drop_weight_keys:
                    if key.startswith(drop_key):
                        del state_dict[key]

            if key.startswith('fpn.lat_layers'):
                transfered_from_yolact = True
                state_dict[key.replace('fpn.', 'fpn_phase_1.')] = state_dict[key]
                del state_dict[key]
            elif key.startswith('fpn.') and key in state_dict:
                transfered_from_yolact = True
                state_dict[key.replace('fpn.', 'fpn_phase_2.')] = state_dict[key]
                del state_dict[key]

        keys_not_exist = []
        keys_not_used = []
        keys_mismatch = []

        for key in list(cur_state_dict.keys()):
            if args is not None and args.drop_weights is not None:
                for drop_key in drop_weight_keys:
                    if key.startswith(drop_key):
                        state_dict[key] = cur_state_dict[key]

            # for compatibility with models without existing modules
            if key not in state_dict:
                keys_not_exist.append(key)
                state_dict[key] = cur_state_dict[key]
            else:
                # check key size mismatches
                if state_dict[key].size() != cur_state_dict[key].size():
                    keys_mismatch.append(key)
                    state_dict[key] = cur_state_dict[key]


        # for compatibility with models with simpler architectures, remove unused weights.
        for key in list(state_dict.keys()):
            if key not in cur_state_dict:
                keys_not_used.append(key)
                del state_dict[key]

        self.load_state_dict(state_dict)

        if not self.training:
            self.create_partial_backbone()
