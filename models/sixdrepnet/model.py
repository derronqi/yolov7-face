import math

import torch
from torch import nn

from .repvgg import get_RepVGG_func_by_name
from . import sixd_utils

class SixDRepNet(nn.Module):
    def __init__(self,
                 backbone_name, backbone_file, deploy,
                 bins=(1, 2, 3, 6),
                 droBatchNorm=nn.BatchNorm2d,
                 pretrained=True,
                 postprocess=True,
                 gpu_id=0):
        super(SixDRepNet, self).__init__()
        self.gpu_id = gpu_id
        self.postprocess = postprocess
        repvgg_fn = get_RepVGG_func_by_name(backbone_name)
        backbone = repvgg_fn(deploy)
        if pretrained:
            checkpoint = torch.load(backbone_file)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k,
                    v in checkpoint.items()}  # strip the names
            backbone.load_state_dict(ckpt)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = backbone.stage0, backbone.stage1, backbone.stage2, backbone.stage3, backbone.stage4
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        last_channel = 0
        for n, m in self.layer4.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                last_channel = m.out_channels

        fea_dim = last_channel

        self.linear_reg = nn.Linear(fea_dim, 6)

    def forward(self, x):

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x= self.gap(x)
        x = torch.flatten(x, 1)
        x = self.linear_reg(x)
        if self.postprocess:
            if self.gpu_id ==-1:
                return sixd_utils.compute_rotation_matrix_from_ortho6d(x, False, self.gpu_id)
            else:
                return sixd_utils.compute_rotation_matrix_from_ortho6d(x, True, self.gpu_id)
        else:
            # use postprocess in tensorRT result
            return x
