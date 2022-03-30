import torch 
import torch.nn as nn

import utils.CLS_SEG_REC_structure as MTL_structure

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence



# Up Task 
def Uptask_CLS_SEG_REC_model(end2end=False, backbone_name='resnet50'):
    aux_params   = dict(pooling='avg', dropout=0.5, activation=None, classes=1)
    model        = MTL_structure.Unet(encoder_name=backbone_name, encoder_weights=None, in_channels=1, classes=1, activation=None, aux_params=aux_params)     
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Learnable Params:', n_parameters)
    return model


# Down Task
class Down_SMART_Net_CLS(torch.nn.Module):
    def __init__(self):
        super(Down_SMART_Net_CLS, self).__init__()        
        
        # Slicewise Feature Extract
        self.encoder = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=1, classes=1, activation=None).encoder
        self.pool    = torch.nn.AdaptiveAvgPool2d(1)
        
        # 3D Classifier - ResNet50 based
        self.LSTM    = nn.LSTM(input_size=2048, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)
        self.fc      = nn.Linear(512*2, 512, True)
        self.relu    = nn.ReLU(True)
        self.drop    = nn.Dropout(p=0.5)
        
        # Head
        self.head    = nn.Linear(512, 1, True)       

    
    def forward(self, x, x_lens):
        cnn_embed_seq = []
        for i in range(x.shape[-1]):
            out = self.encoder(x[..., i])[-1]
            out = self.pool(out)
            out = out.view(out.shape[0], -1)
            cnn_embed_seq.append(out)   
        

        stacked_feat = torch.stack(cnn_embed_seq, dim=1)

        self.LSTM.flatten_parameters()  # For Multi GPU  
        x_packed = pack_padded_sequence(stacked_feat, x_lens.cpu(), batch_first=True, enforce_sorted=False)  # x_len이 cpu int64로 들어가야함!!!
        RNN_out, (h_n, h_c) = self.LSTM(x_packed, None)    # input shape = batch, seq, feature
        
        fc_input = torch.cat([h_n[-1], h_n[-2]], dim=-1) # Due to the Bi-directional
        x = self.linear1(fc_input)
        x = self.relu1(x)  
        x = self.drop1(x)  
        x = self.fc(x)     

        return x     

class Patient_Seg_model(torch.nn.Module):
    def __init__(self, end2end=False, backbone_name='resnet50', freeze_decoder=False):
        super(Patient_Seg_model, self).__init__()
        self.end2end = end2end        
        self.freeze_decoder = freeze_decoder
        print("Freeze == ", self.freeze_decoder)
        self.encoder   = smp.Unet(encoder_name=backbone_name, encoder_weights=None, in_channels=1, classes=1, activation=None).encoder
        self.decoder   = smp.Unet(encoder_name=backbone_name, encoder_weights=None, in_channels=1, classes=1, activation=None).decoder
        
        self.conv1 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1   = nn.BatchNorm3d(16)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2   = nn.BatchNorm3d(16)
        self.relu2 = nn.ReLU()
        
        self.last  = nn.Conv3d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)

    
    def forward(self, x):
        N, C, H, W, D = x.shape
        x = x.permute(4, 0, 1, 2, 3)
        x = torch.reshape(x, [D*N, C, H, W])
        
        if self.end2end:
            features = self.encoder(x)
            x        = self.decoder(*features) 

        else:
            if self.freeze_decoder:
                self.encoder.eval()
                self.decoder.eval()
                with torch.no_grad():
                    features = self.encoder(x)
                    x        = self.decoder(*features)
                

            else :
                self.encoder.eval()
                with torch.no_grad():
                    features = self.encoder(x)

                x = self.decoder(*features)                


        
        new_shape = [D, N] + list(x.shape)[1:]
        x = torch.reshape(x, new_shape)
        x = x.permute(1,2,3,4,0)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.last(x)
        
        return x


######################################################    [Cls] Model                    
def Downtask_Cls_model(pretrained=None, end2end=False, backbone_name='resnet50'):
    model      = Patient_Cls_model(end2end=end2end, backbone_name=backbone_name)

    if pretrained is not None:
        if backbone_name == 'resnet50':
            check_point     = torch.load(pretrained)
            model_dict      = model.state_dict()
            print("Check Before weight = ", model_dict['encoder.conv1.weight'][0])
            pretrained_dict = {k: v for k, v in check_point['model_state_dict'].items() if k in model_dict and 'encoder.' in k}
            model_dict.update(pretrained_dict) 
            model.load_state_dict(model_dict)        
            print("Check After weight = ", model_dict['encoder.conv1.weight'][0])
            print("Succeed Load Pretrained...!")   
        else :
            check_point     = torch.load(pretrained)
            model_dict      = model.state_dict()
            print("Check Before weight = ", model_dict['encoder._conv_stem.weight'][0])
            pretrained_dict = {k: v for k, v in check_point['model_state_dict'].items() if k in model_dict and 'encoder.' in k}
            model_dict.update(pretrained_dict) 
            model.load_state_dict(model_dict)        
            print("Check After weight = ", model_dict['encoder._conv_stem.weight'][0])
            print("Succeed Load Pretrained...!")   

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Learnable Params:', n_parameters)

    return model



######################################################    [Seg] Model
def Downtask_Seg_model(pretrained=None, end2end=False, backbone_name='resnet50', decoder_transfer=False, freeze_decoder=False):
    model      = Patient_Seg_model(end2end=end2end, backbone_name=backbone_name, freeze_decoder=freeze_decoder)

    if pretrained is not None:
        if backbone_name == 'resnet50':
            check_point     = torch.load(pretrained)
            model_dict      = model.state_dict()
            
            # Encoder
            print("Check Before weight = ", model_dict['encoder.conv1.weight'][0])
            pretrained_dict = {k: v for k, v in check_point['model_state_dict'].items() if k in model_dict and 'encoder.' in k}
            model_dict.update(pretrained_dict) 
            model.load_state_dict(model_dict)        
            print("Check After weight = ", model_dict['encoder.conv1.weight'][0])
            print("Succeed Load Pretrained...!")   

            # Decoder
            if decoder_transfer:
                print("Check Before weight = ", model_dict['decoder.blocks.0.conv1.0.weight'][0])
                pretrained_dict = {k: v for k, v in check_point['model_state_dict'].items() if k in model_dict and 'decoder.' in k}
                model_dict.update(pretrained_dict) 
                model.load_state_dict(model_dict)        
                print("Check After weight = ", model_dict['decoder.blocks.0.conv1.0.weight'][0])
                print("Succeed Load Pretrained...!")   

        else :
            check_point     = torch.load(pretrained)
            model_dict      = model.state_dict()
            print("Check Before weight = ", model_dict['encoder._conv_stem.weight'][0])
            pretrained_dict = {k: v for k, v in check_point['model_state_dict'].items() if k in model_dict and 'encoder.' in k}
            model_dict.update(pretrained_dict) 
            model.load_state_dict(model_dict)        
            print("Check After weight = ", model_dict['encoder._conv_stem.weight'][0])
            print("Succeed Load Pretrained...!")  

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Learnable Params:', n_parameters)

    return model





######################################################                    Create Model                        ########################################################

models = {    
    # Up Stream 
    "Up_SMART_Net": {
        "model": Uptask_CLS_SEG_REC_model,
        "pretrained": '/workspace/sunggu/1.Hemorrhage/models/MTL/[UpTASK]Hemo_resnet50_OnlySeg/epoch_700_best_metric_model.pth',
    },    
    
    # Down Stream 
    "Down_SMART_Net_CLS": {
        "model": Downtask_Cls_model,
        "pretrained": '/workspace/sunggu/1.Hemorrhage/models/MTL/[UpTASK]Hemo_resnet50_OnlySeg/epoch_700_best_metric_model.pth',
    },
    "Down_SMART_Net_SEG": {
        "model": Downtask_Seg_model,
        "pretrained": '/workspace/sunggu/1.Hemorrhage/models/MTL/[UpTASK]Hemo_resnet50_OnlySeg/epoch_700_best_metric_model.pth',
    },       
}


def create_model(name, pretrained, end2end, backbone_name, decoder_transfer, freeze_decoder):
    try:
        if pretrained:
            model = models[name]["model"](pretrained=pretrained, end2end=end2end, backbone_name=backbone_name, decoder_transfer=decoder_transfer, freeze_decoder=freeze_decoder)
        else :
            model = models[name]["model"](end2end=end2end, backbone_name=backbone_name)

    except KeyError:
        raise KeyError("Wrong model name `{}`, supported models: {}".format(name, list(model.keys())))

    return model


def create_model(stream, name, pretrained):
    if stream == 'Upstream':
        # Ours
        if name == 'Uptask_Sup_Classifier':
            # model = Supervised_Classifier_Model(pretrained)
            pass
        

        # Previous works
        elif name == 'Uptask_Unsup_AutoEncoder':
            model = Resnet_AutoEncoder()

        elif name == 'Unsupervised_ModelGenesis_Model':
            model = ModelGenesis()

        # elif name == 'Unsupervised_TransVW_Model':
            # model = TransVW()            


    elif stream == 'Downstream':
        if name == '1.General Fracture':
            model = General_Fracture_Model(pretrained)
        elif name == '2.RSNA BoneAge':
            model = RSNA_BoneAge_Model(pretrained)
        elif name == '3.Nodule Detection':
            model = Nodule_Detection_Model(pretrained)                        


    else :
        raise KeyError("Wrong model name `{}`".format(name))
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Learnable Params:', n_parameters)   

    return model