# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel

import io
import json
import zipfile
from mmpretrain.registry import MODELS
from xgboost import XGBClassifier
from pytorch_tabnet import tab_network
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
class VTT6DCPooling(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self, dim=2, hidden_dim = 2048, Table = False, bert_model_name = None, num_classes = 6):
        super(VTT6DCPooling, self).__init__()
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
        self.num_classes = num_classes
        #self.stage = stage # feature
        self.bert_model_name = bert_model_name
        self.bert = BertModel.from_pretrained(self.bert_model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_classes)

        if self.Table:
            self.group_matrix = create_group_matrix(list_groups=[],input_dim=10).to('cuda')
            self.model = tab_network.TabNet(input_dim = 10,
                                            output_dim = num_classes,
                                            group_attention_matrix = self.group_matrix).to('cuda')
            saved_state_dict = self.load_model('/home/user/ZY_Workspace/Django_VTmamba/mmpretrain/tabnet_6/tabnet_model_0.61.zip')
            self.model.load_state_dict(saved_state_dict)
            self.model.eval()
            self.cross_attention = CrossAttention(2048, 24, 64, 64, 4)
            self.cross_attention_text = CrossAttention(2048, 768, 64, 64, 4)


    def forward(self, inputs, inputs_table, inputs_text):    
        #print(inputs_text['input_ids'].shape, inputs_text['attention_mask'].shape, inputs_text['token_type_ids'].shape)
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
                outs_table = (self.model(inputs_table))
            outs_table_feature = outs_table[2].view(-1,24).detach()
            outs_ = self.cross_attention(outs[-1], outs_table_feature) 
            ################################################
            #add bert
            self.bert.eval()
            input_ids = inputs_text['input_ids'].view(-1, 256)
            attention_mask = inputs_text['attention_mask'].view(-1, 256)
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            text_info = outputs[1]#.detach()
            outs = self.cross_attention_text(outs_, text_info) + outs[-1]
            #print(text_info)
            cls_text = self.fc(text_info)  
            ################################################
            #outs = outs[-1]
            feat = {'feat': outs, 'feat_table': outs_table ,'feat_text': cls_text}
            outs = feat
         
        return outs
    
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
    


