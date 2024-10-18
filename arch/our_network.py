import torch 
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers.conv2d_same import Conv2dSame
from timm.layers.norm import LayerNorm2d
# from timm.models.layers.conv2d_same import Conv2dSame
# from timm.models.layers.norm import LayerNorm2d

from typing import Any, Iterator, Mapping
from itertools import chain

import monai
import types
from itertools import islice
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from monai.inferers import SliceInferer
from transformers import BertModel, BertConfig
from monai.networks.nets.vit import ViT
from einops import rearrange
from einops.layers.torch import Rearrange

# from arch.deeplab import ASPP, SeparableConv3d
# from module.Non_Local_block import NONLocalBlock3D
# from sliding_window import sliding_window_inference_cls_output, sliding_window_inference_seg_output


def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def initialize_decoder(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# Modules
class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not (use_batchnorm))
        relu = nn.ReLU(inplace=True)
        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()
        super(Conv2dReLU, self).__init__(conv, bn, relu)

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()
        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class SEG_DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True, attention_type=None, last=False):
        super().__init__()
        self.attention1 = nn.Identity() if last else Attention(attention_type, in_channels=in_channels+skip_channels)
        self.conv1      = Conv2dReLU(in_channels+skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2      = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, scale, input_channels, output_channels, ksize=1):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(input_channels, output_channels*(scale**2), kernel_size=1, stride=1, padding=ksize//2),
            nn.PixelShuffle(upscale_factor=scale)
        )

    def forward(self, input):
        return self.upsample(input)

class REC_Skip_DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True, attention_type=None, last=False):
        super().__init__()
        self.upsample   = UpsampleBlock(scale=2, input_channels=in_channels, output_channels=in_channels)
        self.attention1 = nn.Identity() if last else Attention(attention_type, in_channels=in_channels+skip_channels)
        self.conv1      = Conv2dReLU(in_channels+skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2      = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class REC_DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True, attention_type=None, last=False):
        super().__init__()
        self.upsample   = UpsampleBlock(scale=2, input_channels=in_channels, output_channels=in_channels)
        self.attention1 = nn.Identity() if last else Attention(attention_type, in_channels=in_channels)
        self.conv1      = Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2      = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class Window_Conv2D(nn.Module):    
    '''
    Ref: https://arxiv.org/pdf/1812.00572.pdf
    HU summary  
          [HU threshold]                           [weight / bias]
    softtissue = W:350 L:40 (-135 ~ 215)         W:11.70275 B:-2.53996
    liver      = W:150 L:30 (-45  ~ 105)         W:27.30748 B:-6.52676

    softtissue =  Adapt                          W:4.00000 B:-2.47143
    liver      =  Adapt                          W:9.33333 B:-6.36667
    '''      

    def __init__(self, mode, in_channels=1, w_channels=2):
        super(Window_Conv2D, self).__init__()
        self.w_channels = w_channels
        self.conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=w_channels, kernel_size=1, stride=1, padding=0, bias=True)
        
        if mode == "relu":
            self.act_layer = self.upbound_relu
        elif mode == "sigmoid":
            self.act_layer = self.upbound_sigmoid
        else:
            raise Exception()
        
        # Initialize by xavier_uniform_
        self.init_weight()
        
    def upbound_relu(self, x):
        return torch.minimum(torch.maximum(x, torch.tensor(0)), torch.tensor(1.0))

    def upbound_sigmoid(self, x):
        return 1.0 * torch.sigmoid(x)
                    
    def init_weight(self):
        print("inintializing...!")
        # Range [-1 ~ 1]
        # softtissue
        self.conv_layer.weight.data[0, :, :, :] = 5.850000000000001
        self.conv_layer.bias.data[0]            = 3.310000000000001
        # liver
        self.conv_layer.weight.data[1, :, :, :] = 13.650000000000007
        self.conv_layer.bias.data[1]            = 7.123333333333336
    
    def cal_w_b(self, min, max):
        min = (((min + 1024)/4095) - 0.5) / 0.5
        max = (((max + 1024)/4095) - 0.5) / 0.5
        
        width = max - min
        level = (max + min) / 2.0

        weight = 1/width
        bias = -(1/width) * (level - width/2)
        
        return weight, bias

    def forward(self, x):
        windowed_x = self.conv_layer(x)
        windowed_x = self.act_layer(windowed_x)
        return torch.cat([x, windowed_x], dim=1)
    
    def inference(self, x):
        self.eval()
        with torch.no_grad():
            windowed_x = self.conv_layer(x)
            windowed_x = self.act_layer(windowed_x)
        return torch.cat([x, windowed_x], dim=1)   

class GeneralizedMeanPooling(nn.Module):
    # Ref: Fine-tuning CNN Image Retrieval with No Human Annotation (https://arxiv.org/pdf/1711.02512)
    def __init__(self, p=3, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        self.p   = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def gem(self, x, p=3, eps=1e-6):
        return F.adaptive_avg_pool2d(input=x.clamp(min=eps).pow(p), output_size=1).pow(1./p)

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps).flatten(start_dim=1, end_dim=-1)  # (B, C)

class MaxAvgPooling(nn.Module):
    # We have to adjust the segmentation pred depending on classification pred
    # ResNet50 uses four 2x2 maxpools and 1 global avgpool to extract classification pred. that is the same as 16x16 maxpool and 16x16 avgpool
    def __init__(self):
        super(MaxAvgPooling, self).__init__()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=16, stride=16, padding=0)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=16, stride=16, padding=0)

    def forward(self, x):
        return self.avgpool(self.maxpool(x)).flatten(start_dim=1, end_dim=-1)  # (B, C)

class ViT_Decoder(ViT):
    def __init__(self, *args, **kwargs):
        super(ViT_Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = self.patch_embedding(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x






# 2D: encoder with MTL
class SMART_Net_2D(nn.Module):
    def __init__(self, backbone='resnet50', use_skip=True, pool_type='gem', use_consist=True, roi_size=256):
        super(SMART_Net_2D, self).__init__()
        
        self.backbone    = backbone
        self.use_skip    = use_skip
        self.pool_type   = pool_type
        self.use_consist = use_consist
        
        if backbone == 'resnet-50':
            self.encoder = monai.networks.nets.ResNetFeatures(model_name='resnet50', pretrained=False, spatial_dims=2, in_channels=1)
            self.output_channels = [2048, 1024, 512, 256, 64]
        elif backbone == 'efficientnet-b7':
            self.encoder = monai.networks.nets.EfficientNetBNFeatures(model_name="efficientnet-b7", pretrained=True, spatial_dims=2, in_channels=1)
            self.output_channels = [640, 224, 80, 48, 32]
        elif backbone == 'maxvit-xlarge':
            self.encoder = timm.create_model('maxvit_xlarge_tf_512.in21k_ft_in1k', pretrained=True, features_only=True, img_size=roi_size)
            self.encoder.stem.conv1 = Conv2dSame(1, 192, kernel_size=(3, 3), stride=(2, 2))
            self.output_channels = [1536, 768, 384, 192, 192]
        elif backbone == 'maxvit-small':
            self.encoder = timm.create_model('maxvit_small_tf_512.in1k', pretrained=True, features_only=True, img_size=roi_size)
            self.encoder.stem.conv1 = Conv2dSame(1, 64, kernel_size=(3, 3), stride=(2, 2))
            self.output_channels = [768, 384, 192, 96, 64]            

        # CLS Decoder
        if backbone == 'resnet50' or backbone == 'efficientnet-b7':
            # Pooling -> Flatten -> FC -> ReLU -> Drop -> FC
            self.cls_decoder_block1 = nn.AdaptiveAvgPool2d(output_size=1)
            self.cls_decoder_block2 = Flatten()
            self.cls_decoder_block3 = nn.Linear(in_features=self.output_channels[0], out_features=self.output_channels[0]//2, bias=True)
            self.cls_decoder_block4 = nn.ReLU()
            self.cls_decoder_block5 = nn.Dropout(p=0.3, inplace=False)
        
        else:
            # Pooling -> Norm -> Flatten -> FC -> Tanh -> Drop -> FC
            self.cls_decoder_block1 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=1), LayerNorm2d(num_channels=self.output_channels[0], eps=1e-05, affine=True))
            self.cls_decoder_block2 = Flatten()
            self.cls_decoder_block3 = nn.Linear(in_features=self.output_channels[0], out_features=self.output_channels[0]//2, bias=True)
            self.cls_decoder_block4 = nn.Tanh()
            self.cls_decoder_block5 = nn.Dropout(p=0.3, inplace=False)            

        # SEG Decoder
        self.seg_decoder_block1 = SEG_DecoderBlock(in_channels=self.output_channels[0], skip_channels=self.output_channels[1], out_channels=256, use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block2 = SEG_DecoderBlock(in_channels=256,                     skip_channels=self.output_channels[2], out_channels=128, use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block3 = SEG_DecoderBlock(in_channels=128,                     skip_channels=self.output_channels[3], out_channels=64,  use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block4 = SEG_DecoderBlock(in_channels=64,                      skip_channels=self.output_channels[4], out_channels=32,  use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block5 = SEG_DecoderBlock(in_channels=32,                      skip_channels=0,                       out_channels=16,  use_batchnorm=True, attention_type='scse', last=True)

        # REC Decoder
        Selected_REC_DecoderBlock = REC_Skip_DecoderBlock if use_skip else REC_DecoderBlock
        self.rec_decoder_block1   = Selected_REC_DecoderBlock(in_channels=self.output_channels[0], skip_channels=self.output_channels[1], out_channels=256, use_batchnorm=True, attention_type='scse')
        self.rec_decoder_block2   = Selected_REC_DecoderBlock(in_channels=256,                     skip_channels=self.output_channels[2], out_channels=128, use_batchnorm=True, attention_type='scse')
        self.rec_decoder_block3   = Selected_REC_DecoderBlock(in_channels=128,                     skip_channels=self.output_channels[3], out_channels=64,  use_batchnorm=True, attention_type='scse')
        self.rec_decoder_block4   = Selected_REC_DecoderBlock(in_channels=64,                      skip_channels=self.output_channels[4], out_channels=32,  use_batchnorm=True, attention_type='scse')
        self.rec_decoder_block5   = Selected_REC_DecoderBlock(in_channels=32,                      skip_channels=0,                       out_channels=16,  use_batchnorm=True, attention_type='scse', last=True)

        # Head
        self.cls_head = nn.Linear(in_features=self.output_channels[0]//2, out_features=1, bias=True)
        self.seg_head = nn.Conv2d(in_channels=16,   out_channels=1, kernel_size=3, padding=1)
        self.rec_head = nn.Conv2d(in_channels=16,   out_channels=1, kernel_size=3, padding=1)

        # For consistency loss
        if pool_type == 'gem':
            self.pool_for_consist = GeneralizedMeanPooling(p=3, eps=1e-6) # GeM pooling
        else:
            self.pool_for_consist = MaxAvgPooling()
            
        # Init
        self.initialize()

    def initialize(self):
        initialize_decoder(self.cls_decoder_block1)
        initialize_decoder(self.cls_decoder_block2)
        initialize_decoder(self.cls_decoder_block3)
        initialize_decoder(self.cls_decoder_block4)
        initialize_decoder(self.cls_decoder_block5)               
        initialize_decoder(self.seg_decoder_block1)
        initialize_decoder(self.seg_decoder_block2)
        initialize_decoder(self.seg_decoder_block3)
        initialize_decoder(self.seg_decoder_block4)
        initialize_decoder(self.seg_decoder_block5) 
        initialize_decoder(self.rec_decoder_block1)
        initialize_decoder(self.rec_decoder_block2)
        initialize_decoder(self.rec_decoder_block3)
        initialize_decoder(self.rec_decoder_block4)
        initialize_decoder(self.rec_decoder_block5)     
        initialize_head(self.cls_head)
        initialize_head(self.seg_head)
        initialize_head(self.rec_head)
 
    def forward(self, x):
        # encoder
        skip4, skip3, skip2, skip1, x = self.encoder(x)

        # cls decoder
        cls = self.cls_decoder_block1(x)
        cls = self.cls_decoder_block2(cls)
        cls = self.cls_decoder_block3(cls)
        cls = self.cls_decoder_block4(cls)
        cls = self.cls_decoder_block5(cls)

        # seg decoder
        seg = self.seg_decoder_block1(x,   skip1)
        seg = self.seg_decoder_block2(seg, skip2)
        seg = self.seg_decoder_block3(seg, skip3)
        seg = self.seg_decoder_block4(seg, skip4)        
        seg = self.seg_decoder_block5(seg)

        # rec decoder
        rec = self.rec_decoder_block1(x,   skip1)
        rec = self.rec_decoder_block2(rec, skip2)
        rec = self.rec_decoder_block3(rec, skip3)
        rec = self.rec_decoder_block4(rec, skip4)
        rec = self.rec_decoder_block5(rec)

        # head
        cls = self.cls_head(cls)
        seg = self.seg_head(seg)
        rec = self.rec_head(rec)
        
        if self.use_consist:
            return cls, seg, rec, self.pool_for_consist(seg)
        else:
            return cls, seg, rec
    





# 3D-CLS: 3D operator w/ 2D encoder
class SMART_Net_3D_CLS(nn.Module):
    def __init__(self, transfer_pretrained=None, use_pretrained_encoder=True, freeze_encoder=True, roi_size=512, sw_batch_size=64, spatial_dim=0, backbone='maxvit-xlarge', use_skip=True, pool_type='gem', operator_3d='lstm'):
        super(SMART_Net_3D_CLS, self).__init__()
        self.slice_inferer = SliceInferer(roi_size=(roi_size, roi_size), sw_batch_size=sw_batch_size, spatial_dim=spatial_dim, device=torch.device("cuda"))
        self.operator_3d = operator_3d
        self.freeze_encoder = freeze_encoder

        # 2D Encoder
        self.model_2d = SMART_Net_2D(backbone=backbone, use_skip=use_skip, pool_type=pool_type)
        self.feat_dim = self.model_2d.output_channels[0]
        self.encoder  = self.model_2d.encoder

        # 3D operator
        if operator_3d == 'lstm':
            # FC -> ReLU -> Drop -> FC
            self.LSTM        = nn.LSTM(input_size=self.feat_dim, hidden_size=self.feat_dim, num_layers=3, batch_first=True, bidirectional=True)
            self.linear_lstm = nn.Linear(self.feat_dim*2, self.feat_dim, True)
            self.relu_lstm   = nn.ReLU()
        elif operator_3d == 'bert':
            # FC -> Tanh -> Drop -> FC
            self.config = BertConfig.from_pretrained("bert-base-uncased")
            self.config.hidden_size=self.feat_dim             # hidden_size를 self.feat_dim으로 설정
            self.config.num_hidden_layers=12                  # BERT base에서 사용하는 레이어 수
            self.config.num_attention_heads=8                 # 640 hidden size에 맞게 8 헤드 사용
            self.config.intermediate_size=self.feat_dim*4     # FFN 레이어의 크기 (보통 hidden_size * 4)
            self.BERT = BertModel(config=self.config)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.feat_dim)) # BERT 안에서 pos_embed 더해줌.

        # Head
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.feat_dim, 1, True))

        # Init
        if (transfer_pretrained is not None) and (use_pretrained_encoder):
            self.load_pretrained_encoder(transfer_pretrained)

    def load_pretrained_encoder(self, transfer_pretrained):
        print("Load State Dict...!")
        check_layer = list(self.encoder.state_dict().keys())[0]
        print("Original weight = ", self.encoder.state_dict()[check_layer][0].mean())
        checkpoint = torch.load(transfer_pretrained, map_location=torch.device('cpu'))['model_state_dict'] # 초반 EfficientNet + Focal + Tversky
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        checkpoint = {k.replace('encoder.', ''): v for k, v in checkpoint.items() if 'encoder.' in k}
        self.encoder.load_state_dict(checkpoint)
        print("Updated weight = ", self.encoder.state_dict()[check_layer][0].mean())

    def feat_cls_extract(self, x):
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                last_feat = self.encoder.forward(x)[-1]
                last_feat = F.adaptive_avg_pool2d(last_feat, output_size=1)
        else:
            last_feat = self.encoder.forward(x)[-1]
            last_feat = F.adaptive_avg_pool2d(last_feat, output_size=1)
        return last_feat # (B, C, 1, 1)

    def forward(self, x, x_lens):
        # CNN feature extraction
        sequenced_feat = self.slice_inferer(x, self.feat_cls_extract) # output: [B, C, Depth, 1, 1]
        sequenced_feat = sequenced_feat.flatten(start_dim=2) # output: [B, C, Depth]
        sequenced_feat = sequenced_feat.permute(0, 2, 1)     # output: [B, Depth, C]

        # Squential representation
        if self.operator_3d == 'lstm':
            self.LSTM.flatten_parameters()  # For Multi GPU  
            x_packed = pack_padded_sequence(sequenced_feat, x_lens.cpu(), batch_first=True, enforce_sorted=False)  # x_len이 cpu int64로 들어가야함!!!
            RNN_out, (h_n, h_c) = self.LSTM(x_packed, None)    # input shape must be [batch, seq, feat_dim]
            fc_output = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1) # Due to the Bi-directional
            fc_output = self.linear_lstm(fc_output)
            fc_output = self.relu_lstm(fc_output)
        elif self.operator_3d == 'bert':
            cls_tokens     = self.cls_token.expand(sequenced_feat.shape[0], -1, -1)
            sequenced_feat = torch.cat((cls_tokens, sequenced_feat), dim=1)  # cls token 추가
            attention_mask = torch.ones(sequenced_feat.shape[:2], dtype=torch.long).to(sequenced_feat.device) # x_lens을 이용하여 attention_mask 생성
            for i, x_len in enumerate(x_lens):
                attention_mask[i, x_len:] = 0 # 여기 디버깅 필요.
            fc_output = self.BERT(inputs_embeds=sequenced_feat, attention_mask=attention_mask).pooler_output # inputs_embeds shape must be (batch_size, sequence_length, hidden_size)

        # HEAD
        fc_output = self.head(fc_output)
    
        return fc_output     


    # def forward(self, x, x_lens):
    #     B, C, D, H, W = x.shape
    #     slice_feat_list = []
    #     for i in range(D):
    #         slice_feat = self.encoder(x[:, :, i, :, :])[-1]
    #         slice_feat = F.adaptive_avg_pool2d(slice_feat, output_size=1)
    #         # slice_feat = self.feat_cls_extract(x[:, :, i, :, :])
    #         slice_feat_list.append(slice_feat)
        
    #     stacked_slice_feat = torch.stack(slice_feat_list, dim=2) # output: [B, C, Depth, 1, 1]
    #     sequenced_feat = stacked_slice_feat.flatten(start_dim=2)    # output: [B, C, Depth]
    #     sequenced_feat = sequenced_feat.permute(0, 2, 1)            # output: [B, Depth, C]

    #     # CNN feature extraction
    #     # sequenced_feat = self.slice_inferer(x, self.feat_cls_extract) # output: [B, C, Depth, 1, 1]
    #     # sequenced_feat = sequenced_feat.flatten(start_dim=2) # output: [B, C, Depth]
    #     # sequenced_feat = sequenced_feat.permute(0, 2, 1)     # output: [B, Depth, C]

    #     # Squential representation
    #     if self.operator_3d == 'lstm':
    #         self.LSTM.flatten_parameters()  # For Multi GPU  
    #         x_packed = pack_padded_sequence(sequenced_feat, x_lens.cpu(), batch_first=True, enforce_sorted=False)  # x_len이 cpu int64로 들어가야함!!!
    #         RNN_out, (h_n, h_c) = self.LSTM(x_packed, None)    # input shape must be [batch, seq, feat_dim]
    #         fc_output = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1) # Due to the Bi-directional
    #         fc_output = self.linear_lstm(fc_output)
    #         fc_output = self.relu_lstm(fc_output)
    #     elif self.operator_3d == 'bert':
    #         cls_tokens     = self.cls_token.expand(sequenced_feat.shape[0], -1, -1)
    #         sequenced_feat = torch.cat((cls_tokens, sequenced_feat), dim=1)  # cls token 추가
    #         attention_mask = torch.ones(sequenced_feat.shape[:2], dtype=torch.long).to(sequenced_feat.device) # x_lens을 이용하여 attention_mask 생성
    #         for i, x_len in enumerate(x_lens):
    #             attention_mask[i, x_len:] = 0 # 여기 디버깅 필요.
    #         fc_output = self.BERT(inputs_embeds=sequenced_feat, attention_mask=attention_mask).pooler_output # inputs_embeds shape must be (batch_size, sequence_length, hidden_size)

    #     # HEAD
    #     fc_output = self.head(fc_output)
    
    #     return fc_output    
    

# 3D-SEG: 3D operator w/ 2D encoder
class SMART_Net_3D_SEG(nn.Module):
    def __init__(self, transfer_pretrained=None, use_pretrained_encoder=True, use_pretrained_decoder=False, freeze_encoder=True, freeze_decoder=False, roi_size=512, sw_batch_size=64, spatial_dim=0, backbone='maxvit-xlarge', use_skip=True, pool_type='gem', operator_3d='3d_cnn'):
        super(SMART_Net_3D_SEG, self).__init__()
        self.roi_size = roi_size
        self.slice_inferer = SliceInferer(roi_size=(roi_size, roi_size), sw_batch_size=sw_batch_size, spatial_dim=spatial_dim, device=torch.device("cuda"))
        self.operator_3d = operator_3d
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder

        # 2D Encoder
        self.model_2d = SMART_Net_2D(backbone=backbone, use_skip=use_skip, pool_type=pool_type)
        self.feat_dim = self.model_2d.output_channels
        self.encoder  = self.model_2d.encoder

        # 2D SEG Decoder
        self.seg_decoder1 = SEG_DecoderBlock(in_channels=self.feat_dim[0], skip_channels=self.feat_dim[1], out_channels=256, use_batchnorm=True, attention_type='scse')
        self.seg_decoder2 = SEG_DecoderBlock(in_channels=256,              skip_channels=self.feat_dim[2], out_channels=128, use_batchnorm=True, attention_type='scse')
        self.seg_decoder3 = SEG_DecoderBlock(in_channels=128,              skip_channels=self.feat_dim[3], out_channels=64,  use_batchnorm=True, attention_type='scse')
        self.seg_decoder4 = SEG_DecoderBlock(in_channels=64,               skip_channels=self.feat_dim[4], out_channels=32,  use_batchnorm=True, attention_type='scse')
        self.seg_decoder5 = SEG_DecoderBlock(in_channels=32,               skip_channels=0,                out_channels=16,  use_batchnorm=True, attention_type='scse', last=True)

        # 3D operator
        if self.operator_3d == '3d_cnn':
            self.conv1 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
            self.bn1   = nn.BatchNorm3d(16)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
            self.bn2   = nn.BatchNorm3d(16)
            self.relu2 = nn.ReLU()
        elif self.operator_3d == '3d_vit':
            self.img_size    = (64, roi_size, roi_size)
            self.patch_size  = (4, roi_size//16, roi_size//16)
            self.feat_size   = (self.img_size[0]//self.patch_size[0], self.img_size[1]//self.patch_size[1], self.img_size[2]//self.patch_size[2])
            self.in_channels = 16
            self.vit = ViT_Decoder(in_channels=self.in_channels,
                                   hidden_size=768,
                                   img_size=self.img_size,
                                   patch_size=self.patch_size,
                                   pos_embed="perceptron",
                                   spatial_dims=3,
                                   classification=True)
            self.vit_decoder = nn.Linear(768, self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * self.in_channels, bias=True)  # patch_size[0]*patch_size[1]*patch_size[2]*in_channels
        
        # Head
        self.head = nn.Conv3d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)

        # Init
        if (transfer_pretrained is not None) and (use_pretrained_encoder):
            self.load_pretrained_encoder(transfer_pretrained)
        
        if (transfer_pretrained is not None) and (use_pretrained_decoder):
            self.load_pretrained_decoder(transfer_pretrained)

    def load_pretrained_encoder(self, transfer_pretrained):
        print("Load State Dict...!")
        check_layer = list(self.encoder.state_dict().keys())[0]
        print("Original weight = ", self.encoder.state_dict()[check_layer][0].mean())
        checkpoint = torch.load(transfer_pretrained, map_location=torch.device('cpu'))['model_state_dict'] # 초반 EfficientNet + Focal + Tversky
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        checkpoint = {k.replace('encoder.', ''): v for k, v in checkpoint.items() if 'encoder.' in k}
        self.encoder.load_state_dict(checkpoint)
        print("Updated weight  = ", self.encoder.state_dict()[check_layer][0].mean())

    def load_pretrained_decoder(self, transfer_pretrained):
        print("Load State Dict...!")
        check_layer = list(self.seg_decoder1.state_dict().keys())[0]
        print("Original weight =", self.seg_decoder1.state_dict()[check_layer][0].mean())
        checkpoint = torch.load(transfer_pretrained, map_location=torch.device('cpu'))['model_state_dict']
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        def load_decoder_block(block, block_num):
            block_dict = {k.replace(f'seg_decoder_block{block_num}.', ''): v for k, v in checkpoint.items() if f'seg_decoder_block{block_num}.' in k}
            block.load_state_dict(block_dict)
        for i in range(1, 6):
            load_decoder_block(getattr(self, f'seg_decoder{i}'), i)
        print("Updated weight  =", self.seg_decoder1.state_dict()[check_layer][0].mean())

    def feat_seg_extract(self, x):
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                skip4, skip3, skip2, skip1, x = self.encoder.forward(x)
        else:
            skip4, skip3, skip2, skip1, x = self.encoder.forward(x)
        
        if self.freeze_decoder:
            self.seg_decoder1.eval()
            self.seg_decoder2.eval()
            self.seg_decoder3.eval()
            self.seg_decoder4.eval()
            self.seg_decoder5.eval()
            with torch.no_grad():            
                seg = self.seg_decoder1(x,   skip1)
                seg = self.seg_decoder2(seg, skip2)
                seg = self.seg_decoder3(seg, skip3)
                seg = self.seg_decoder4(seg, skip4)        
                seg = self.seg_decoder5(seg)
        else:
            seg = self.seg_decoder1(x,   skip1)
            seg = self.seg_decoder2(seg, skip2)
            seg = self.seg_decoder3(seg, skip3)
            seg = self.seg_decoder4(seg, skip4)        
            seg = self.seg_decoder5(seg)
        
        return seg


    def forward(self, x):
        # cnn feature extraction
        stacked_feat = self.slice_inferer(x, self.feat_seg_extract) # output shape = [B, C(=16), D, H, W]
        
        # Squential representation
        if self.operator_3d == '3d_cnn':
            output = self.conv1(stacked_feat)  #  input = (B, C, D, H, W)
            output = self.bn1(output)
            output = self.relu1(output)
            output = self.conv2(output)
            output = self.bn2(output)
            output = self.relu2(output)
        elif self.operator_3d == '3d_vit':
            output = self.vit(stacked_feat)    # output = b (d h w) (p1 p2 p3 c)
            output = self.vit_decoder(output)
            output = output[:, 1:, :]   # remove cls token
            # (B, L, patch_size0*patch_size1*patch_size2*in_channels) -> (B, d, h, w, patch_size0, patch_size1, patch_size2, in_channels)
            output = output.view(output.shape[0], self.feat_size[0], self.feat_size[1], self.feat_size[2], self.patch_size[0], self.patch_size[1], self.patch_size[2], self.in_channels)
            output = output.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
            output = output.view(output.shape[0], self.in_channels, self.feat_size[0]*self.patch_size[0], self.feat_size[1]*self.patch_size[1], self.feat_size[2]*self.patch_size[2])

        # Head
        output = self.head(output)
        return output






# 3D - 331
class SCSEModule_3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv3d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class Attention_3D(nn.Module):
    def __init__(self, name, **params):
        super().__init__()
        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule_3D(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)

class Conv3dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not (use_batchnorm))
        relu = nn.ReLU(inplace=True)
        if use_batchnorm:
            bn = nn.BatchNorm3d(out_channels)
        else:
            bn = nn.Identity()
        super(Conv3dReLU, self).__init__(conv, bn, relu)

class SEG_DecoderBlock_3D_331(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True, attention_type=None, last=False):
        super().__init__()
        self.attention1 = nn.Identity() if last else Attention_3D(attention_type, in_channels=in_channels+skip_channels)
        self.conv1      = Conv3dReLU(in_channels+skip_channels, out_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0), use_batchnorm=use_batchnorm)
        self.conv2      = Conv3dReLU(out_channels, out_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0), use_batchnorm=use_batchnorm)        
        self.attention2 = Attention_3D(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=(2, 2, 1), mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class PixelShuffle3d_331(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_height, in_width, in_depth = input.size()
        nOut = channels // self.scale ** 2

        out_depth  = in_depth
        out_height = in_height * self.scale
        out_width  = in_width  * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, 1, in_height, in_width, in_depth)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_height, out_width, out_depth)

class UpsampleBlock_3D_331(nn.Module):
    def __init__(self, scale, input_channels, output_channels, ksize=1):
        super(UpsampleBlock_3D_331, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv3d(input_channels, output_channels*(scale**2), kernel_size=1, stride=1, padding=ksize//2),
            PixelShuffle3d_331(scale=scale)
        )

    def forward(self, input):
        return self.upsample(input)

class REC_Skip_DecoderBlock_3D_331(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True, attention_type=None, last=False):
        super().__init__()
        self.upsample   = UpsampleBlock_3D_331(scale=2, input_channels=in_channels, output_channels=in_channels)
        self.attention1 = nn.Identity() if last else Attention_3D(attention_type, in_channels=in_channels+skip_channels)
        self.conv1      = Conv3dReLU(in_channels+skip_channels, out_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0), use_batchnorm=use_batchnorm)
        self.conv2      = Conv3dReLU(out_channels, out_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0), use_batchnorm=use_batchnorm)        
        self.attention2 = Attention_3D(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class EfficientNetB5_UNet_MTL_CLS_SEG_REC_25D_331_CLS(nn.Module):
    def __init__(self):
        super(EfficientNetB5_UNet_MTL_CLS_SEG_REC_25D_331_CLS, self).__init__()

        # Encoder
        self.encoder = monai.networks.nets.EfficientNetBNFeatures("efficientnet-b5", spatial_dims=3, in_channels=1, num_classes=1)

        # CLS Decoder (ref: https://github.com/Project-MONAI/MONAI/blob/a86c0e01555727bd848275c7eabc5b9dc26da73d/monai/networks/nets/efficientnet.py#L229)
        self.cls_decoder_block1 = nn.Conv3d(512, 2048, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.cls_decoder_block2 = nn.BatchNorm3d(2048, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.cls_decoder_block3 = monai.networks.blocks.activation.MemoryEfficientSwish()
        self.cls_decoder_block4 = nn.AdaptiveAvgPool3d(output_size=1)
        self.cls_decoder_block5 = Flatten()
        self.cls_decoder_block6 = nn.Dropout(p=0.5, inplace=False)

        # SEG Decoder
        self.seg_decoder_block1 = SEG_DecoderBlock_3D_331(in_channels=512, skip_channels=176, out_channels=256, use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block2 = SEG_DecoderBlock_3D_331(in_channels=256, skip_channels=64,  out_channels=128, use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block3 = SEG_DecoderBlock_3D_331(in_channels=128, skip_channels=40,  out_channels=64,  use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block4 = SEG_DecoderBlock_3D_331(in_channels=64,  skip_channels=24,  out_channels=32,  use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block5 = SEG_DecoderBlock_3D_331(in_channels=32,  skip_channels=0,   out_channels=16,  use_batchnorm=True, attention_type='scse', last=True)
        
        # REC Decoder
        self.rec_decoder_block1 = REC_Skip_DecoderBlock_3D_331(in_channels=512, skip_channels=176, out_channels=256, use_batchnorm=True, attention_type='scse')
        self.rec_decoder_block2 = REC_Skip_DecoderBlock_3D_331(in_channels=256, skip_channels=64,  out_channels=128, use_batchnorm=True, attention_type='scse')
        self.rec_decoder_block3 = REC_Skip_DecoderBlock_3D_331(in_channels=128, skip_channels=40,  out_channels=64,  use_batchnorm=True, attention_type='scse')
        self.rec_decoder_block4 = REC_Skip_DecoderBlock_3D_331(in_channels=64,  skip_channels=24,  out_channels=32,  use_batchnorm=True, attention_type='scse')
        self.rec_decoder_block5 = REC_Skip_DecoderBlock_3D_331(in_channels=32,  skip_channels=0,   out_channels=16,  use_batchnorm=True, attention_type='scse', last=True)

        # Head
        self.cls_head = nn.Linear(in_features=2048, out_features=3, bias=True)
        self.seg_head = nn.Conv3d(in_channels=16,   out_channels=1, kernel_size=(3, 3, 1), padding=(1, 1, 0))
        self.rec_head = nn.Conv3d(in_channels=16,   out_channels=1, kernel_size=(3, 3, 1), padding=(1, 1, 0))
        
        # Init
        self.modify_kernel_size(self.encoder)

    def modify_kernel_size(self, module):
        for child_name, child in module.named_children():
            if isinstance(child, nn.Conv3d) and child.kernel_size == (3, 3, 3) and child.stride == (2, 2, 2):
                if child.groups is not None:
                    new = nn.Conv3d(child.in_channels, child.out_channels, kernel_size=(3, 3, 1), stride=(2, 2, 1), groups=child.groups, bias=child.bias is not None)
                    setattr(module, child_name, new)
                else:  
                    new = nn.Conv3d(child.in_channels, child.out_channels, kernel_size=(3, 3, 1), stride=(2, 2, 1), bias=child.bias is not None)
                    setattr(module, child_name, new)

            if isinstance(child, nn.Conv3d) and child.kernel_size == (5, 5, 5) and child.stride == (2, 2, 2):
                if child.groups is not None:
                    new = nn.Conv3d(child.in_channels, child.out_channels, kernel_size=(5, 5, 1), stride=(2, 2, 1), groups=child.groups, bias=child.bias is not None)
                    setattr(module, child_name, new)
                else:  
                    new = nn.Conv3d(child.in_channels, child.out_channels, kernel_size=(5, 5, 1), stride=(2, 2, 1), bias=child.bias is not None)
                    setattr(module, child_name, new)

            if isinstance(child, nn.Conv3d) and child.kernel_size == (3, 3, 3) and child.stride == (1, 1, 1):
                if child.groups is not None:
                    new = nn.Conv3d(child.in_channels, child.out_channels, kernel_size=(3, 3, 1), stride=(1, 1, 1), groups=child.groups, bias=child.bias is not None)
                    setattr(module, child_name, new)
                else:  
                    new = nn.Conv3d(child.in_channels, child.out_channels, kernel_size=(3, 3, 1), stride=(1, 1, 1), bias=child.bias is not None)
                    setattr(module, child_name, new)

            if isinstance(child, nn.Conv3d) and child.kernel_size == (5, 5, 5) and child.stride == (1, 1, 1):
                if child.groups is not None:
                    new = nn.Conv3d(child.in_channels, child.out_channels, kernel_size=(5, 5, 1), stride=(1, 1, 1), groups=child.groups, bias=child.bias is not None)
                    setattr(module, child_name, new)
                else:  
                    new = nn.Conv3d(child.in_channels, child.out_channels, kernel_size=(5, 5, 1), stride=(1, 1, 1), bias=child.bias is not None)
                    setattr(module, child_name, new)

            if isinstance(child, nn.ConstantPad3d) and child.padding == (1, 2, 1, 2, 1, 2):
                new = nn.ConstantPad3d(padding=(0, 0)+child.padding[2:], value=0.0)
                setattr(module, child_name, new)

            if isinstance(child, nn.ConstantPad3d) and child.padding == (0, 1, 0, 1, 0, 1):
                new = nn.ConstantPad3d(padding=(0, 0)+child.padding[2:], value=0.0)
                setattr(module, child_name, new)

            if isinstance(child, nn.ConstantPad3d) and child.padding == (1, 1, 1, 1, 1, 1):
                new = nn.ConstantPad3d(padding=(0, 0)+child.padding[2:], value=0.0)
                setattr(module, child_name, new)                

            if isinstance(child, nn.ConstantPad3d) and child.padding == (2, 2, 2, 2, 2, 2):
                new = nn.ConstantPad3d(padding=(0, 0)+child.padding[2:], value=0.0)
                setattr(module, child_name, new)

            # 재귀적으로 모든 하위 레이어를 검사합니다.
            self.modify_kernel_size(child)

    def forward(self, x):
        # encoder
        skip4, skip3, skip2, skip1, x = self.encoder(x)

        # cls decoder
        cls = self.cls_decoder_block1(x)
        cls = self.cls_decoder_block2(cls)
        cls = self.cls_decoder_block3(cls)
        cls = self.cls_decoder_block4(cls)
        cls = self.cls_decoder_block5(cls)
        cls = self.cls_decoder_block6(cls)

        # seg decoder
        seg = self.seg_decoder_block1(x,   skip1)
        seg = self.seg_decoder_block2(seg, skip2)
        seg = self.seg_decoder_block3(seg, skip3)
        seg = self.seg_decoder_block4(seg, skip4)        
        seg = self.seg_decoder_block5(seg)

        # rec decoder
        rec = self.rec_decoder_block1(x,   skip1)
        rec = self.rec_decoder_block2(rec, skip2)
        rec = self.rec_decoder_block3(rec, skip3)
        rec = self.rec_decoder_block4(rec, skip4)
        rec = self.rec_decoder_block5(rec)

        # head
        cls = self.cls_head(cls)
        seg = self.seg_head(seg)
        rec = self.rec_head(rec)
        
        return cls, seg, rec