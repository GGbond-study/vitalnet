# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tab_transformer_pytorch import TabTransformer

import io
import json
import zipfile
from mmpretrain.registry import MODELS
from pytorch_tabnet.utils import create_group_matrix


class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        
        self.proj_q1 = nn.Linear(in_dim1, k_dim, bias=True)
        self.proj_k2 = nn.Linear(in_dim2, k_dim, bias=True)
        self.proj_v2 = nn.Linear(in_dim2, v_dim, bias=True)
        self.proj_o = nn.Linear(v_dim, in_dim1)
        
    def forward(self, x1, x2, mask=None):
        batch_size, in_dim1 = x1.size()
        
        q1 = self.proj_q1(x1)#32x64
        k2 = self.proj_k2(x2)#32x64
        v2 = self.proj_v2(x2)#32x64
        
        attn = torch.matmul(q1, k2.T) / 8**0.5
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2)
        output = self.proj_o(output)
        
        return output
    
class MutilheadCrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        
        self.proj_q1 = nn.Linear(in_dim1, k_dim*self.num_heads, bias=True)
        self.proj_k2 = nn.Linear(in_dim2, k_dim*self.num_heads, bias=True)
        self.proj_v2 = nn.Linear(in_dim2, v_dim*self.num_heads, bias=True)
        self.proj_o = nn.Linear(v_dim*self.num_heads, in_dim1)
        
    def forward(self, x1, x2, mask=None):
        batch_size, in_dim1 = x1.size()
        
        q1 = self.proj_q1(x1)#32x512
        k2 = self.proj_k2(x2)#32x512
        v2 = self.proj_v2(x2)#32x512
        
        attn = torch.matmul(q1, k2.T) / 8**0.5
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2)
        output = self.proj_o(output)
        
        return output




@MODELS.register_module()
class TABTRANSPooling(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self, dim=2, hidden_dim = 2048, Table = False, stage = 'probe'):
        super(TABTRANSPooling, self).__init__()
        assert dim in [1, 2, 3], 'GlobalAveragePooling dim only support ' \
            f'{1, 2, 3}, get {dim} instead.'
        if dim == 1:
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif dim == 2:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.Table = Table
        self.hidden_dim = hidden_dim
        self.stage = stage # feature

        if self.Table:
            self.group_matrix = create_group_matrix(list_groups=[],input_dim=10).to('cuda')
            self.model = TabTransformer(
                            categories = (2,2),      # tuple containing the number of unique values within each category
                            num_continuous = 8,                # number of continuous values
                            dim = 32,                           # dimension, paper set at 32
                            dim_out = 6,                        # binary prediction, but could be anything
                            depth = 6,                          # depth, paper recommended 6
                            heads = 8,                          # heads, paper recommends 8
                            attn_dropout = 0.1,                 # post-attention dropout
                            ff_dropout = 0.1,                   # feed forward dropout
                            mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
                            mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
                            continuous_mean_std = None # (optional) - normalize the continuous values before layer norm
                            )
            self.model.eval()
            self.cross_attention = CrossAttention(2048, 72, 64, 64, 4)


    def forward(self, inputs, inputs_table):    
        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        if self.Table :
            self.model.eval()
            with torch.no_grad():
                inputs_table = inputs_table.to('cuda')
                ca = inputs_table[:,1:3]
                na = torch.cat([inputs_table[:, :1], inputs_table[:, 3:]], dim=1)
                outs_table, feat_table = (self.model(ca.to(torch.int), na, return_feat = True))
            outs = self.cross_attention(outs[-1], feat_table) + outs[-1]
            #outs = outs[-1]
            #feat = {'feat': outs, 'feat_table': outs_table} #'feat_xgb': outs_xgb}
            #outs = feat
         
        return [outs]
    
    def load_model(self,filepath):
        try:
            with zipfile.ZipFile(filepath) as z:
                with z.open("model_params.json") as f:
                    loaded_params = json.load(f)
                    loaded_params["init_params"]["device_name"] = 'cuda'
                with z.open("network.pt") as f:
                    try:
                        saved_state_dict = torch.load(f, map_location='cuda')
                    except io.UnsupportedOperation:
                        # In Python <3.8.2, the returned file object is not seekable
                        saved_state_dict = torch.load(
                            io.BytesIO(f.read()),
                            map_location='cuda',
                        )
        except KeyError:
            raise KeyError("Your zip file is missing at least one component")
        except zipfile.BadZipFile:
            raise ValueError("The provided file is not a valid ZIP file")
        except EOFError:
            raise ValueError("The 'network.pt' file is incomplete or corrupted")
        return saved_state_dict
    


