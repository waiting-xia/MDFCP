import warnings
warnings.filterwarnings('ignore')

# mmcv、mmengin测试代码
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule
from mmengine.model import constant_init
from mmengine.model.weight_init import trunc_normal_init, normal_init

# mamba测试代码

# dcnv3测试代码

import pkg_resources
dcn_version = float(pkg_resources.get_distribution('DCNv3').version)





