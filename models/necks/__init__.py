# Copyright (c) OpenMMLab. All rights reserved.
from .beitv2_neck import BEiTV2Neck
from .cae_neck import CAENeck
from .densecl_neck import DenseCLNeck
from .gap import GlobalAveragePooling
from .gem import GeneralizedMeanPooling
from .hr_fuse import HRFuseScales
from .itpn_neck import iTPNPretrainDecoder
from .linear_neck import LinearNeck
from .mae_neck import ClsBatchNormNeck, MAEPretrainDecoder
from .milan_neck import MILANPretrainDecoder
from .mixmim_neck import MixMIMPretrainDecoder
from .mocov2_neck import MoCoV2Neck
from .nonlinear_neck import NonLinearNeck
from .simmim_neck import SimMIMLinearDecoder
from .spark_neck import SparKLightDecoder
from .swav_neck import SwAVNeck
from .vt_neck import VTPooling
from .vtt_neck import VTTPooling
from .vtt_new_neck import NVTTPooling
from .vtt_dc_neck import VTTDPooling
from .vtt_neck_cat import VTTCPooling
from .vt_neck_copy import VT6Pooling
from .vtt_neck_6 import VTT6Pooling
from .vtt_neck_6_dc import VTT6DCPooling
from .vtt_neck_all import VTTAPooling
from .neck_feature import FPooling
from .neck_cbam import CBAMPooling
from .neck_concat import CATPooling
from .vt_neck_trans import TABTRANSPooling
from .vtt_neck_clip import CLIPPooling

__all__ = [
    'GlobalAveragePooling',
    'GeneralizedMeanPooling',
    'HRFuseScales',
    'LinearNeck',
    'BEiTV2Neck',
    'CAENeck',
    'DenseCLNeck',
    'MAEPretrainDecoder',
    'ClsBatchNormNeck',
    'MILANPretrainDecoder',
    'MixMIMPretrainDecoder',
    'MoCoV2Neck',
    'NonLinearNeck',
    'SimMIMLinearDecoder',
    'SwAVNeck',
    'iTPNPretrainDecoder',
    'SparKLightDecoder',
    'VTPooling',
    'VTTPooling',
    'NVTTPooling',
    'VTTDPooling',
    'VTTCPooling',
    'VT6Pooling',
    'VTT6Pooling',
    'VTT6DCPooling',
    'VTTAPooling',
    'FPooling',
    'CBAMPooling',
    'CATPooling',
    'TABTRANSPooling',
    'CLIPPooling'
]
