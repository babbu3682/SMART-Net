import torch.nn as nn
import torch

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)
    
#         print("0# =", x.shape)   0# = torch.Size([5, 1024, 2, 40, 40])
        # Squash samples and timesteps into a single axis
        batch, channel, time, height, width = x.shape
        
        x_reshape = x.contiguous().view(batch*time, channel, height, width)  # (samples * timesteps, input_size)
#         print("2# =", x_reshape.shape)  torch.Size([10, 1024, 40, 40])

        y = self.module(x_reshape)
    
#         print("1", y.shape) [10, 512, 40, 40])
#         print("self.module.out_channels", self.module.out_channels) 512

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(batch, self.module.out_channels, time, height, width)  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
            
        return y 
    

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM2d(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM2d, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)  
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
#         print("input_tensor.shape", input_tensor.shape)
        # (b, c, t, h, w) -> (b, t, c, h, w)  
        input_tensor = input_tensor.permute(0, 2, 1, 3, 4)
        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))
        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            # (b, t, c, h, w) -> (b, c, t, h, w)
            layer_output_list = layer_output_list[-1].permute(0, 2, 1, 3, 4)
            last_state_list = last_state_list[-1]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param





# ########################################################################################################################
# ##################### 모델 ##########################
# ########################################################################################################################
# class Flatten(torch.nn.Module):
#     def forward(self, x):
#         return x.view(x.shape[0], -1)

# class conv_block(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(conv_block,self).__init__()
#         self.convlstm1 = ConvLSTM2d(input_dim=ch_in, hidden_dim=ch_out, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True, return_all_layers=False)
#         self.norm1 = nn.BatchNorm3d(ch_out)
#         self.relu1 = nn.ReLU(inplace=True)
        
#         self.convlstm2 = ConvLSTM2d(input_dim=ch_out, hidden_dim=ch_out, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True, return_all_layers=False)
#         self.norm2 = nn.BatchNorm3d(ch_out)
#         self.relu2 = nn.ReLU(inplace=True)
    
#     def forward(self,x):
#         x = self.convlstm1(x)
#         x = self.norm1(x[0])
#         x = self.relu1(x)
#         # x = self.convlstm2(x)
#         # x = self.norm2(x[0])
#         # x = self.relu2(x)
#         return x
    
# class up_conv(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(up_conv,self).__init__()
#         self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
#         self.timedistributed = TimeDistributed(module=nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, padding=1, bias=True), batch_first=True)
#         self.norm = nn.BatchNorm3d(ch_out)
#         self.relu = nn.ReLU(inplace=True)
        

#     def forward(self,x):
#         x = self.up(x)
# #         print("check1",x.shape) check1 torch.Size([5, 1024, 2, 40, 40])
#         x = self.timedistributed(x)
# #         print("check2",x.shape) torch.Size([5, 512, 2, 40, 40])
#         x = self.norm(x)
#         x = self.relu(x)
#         return x
    
    
# class U_Net(nn.Module):
#     def __init__(self, img_ch=1, output_ch=1):
#         super(U_Net,self).__init__()
        
#         self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

#         self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
#         self.Conv2 = conv_block(ch_in=64, ch_out=128)
#         self.Conv3 = conv_block(ch_in=128, ch_out=256)
#         self.Conv4 = conv_block(ch_in=256, ch_out=512)
#         self.Conv5 = conv_block(ch_in=512, ch_out=1024)

#         self.Up5 = up_conv(ch_in=1024, ch_out=512)
#         self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

#         self.Up4 = up_conv(ch_in=512,ch_out=256)
#         self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
#         self.Up3 = up_conv(ch_in=256,ch_out=128)
#         self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
#         self.Up2 = up_conv(ch_in=128,ch_out=64)
#         self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

#         self.Conv_1x1 = nn.Conv3d(64, output_ch, kernel_size=1, stride=1, padding=0)
        
#         self.linear = nn.Sequential(nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(1024, 1, bias=True))

#     def forward(self, x):
#         # encoding path
# #         print("x = " ,x.shape) torch.Size([5, 1, 16, 320, 320])
#         x1 = self.Conv1(x)
# #         print("1 = " ,x1.shape) 1 =  torch.Size([5, 64, 16, 320, 320])
#         x2 = self.Maxpool(x1)
# #         print("2 = " ,x2.shape) 2 =  torch.Size([5, 64, 8, 160, 160])
#         x2 = self.Conv2(x2)
# #         print("3 = " ,x2.shape) 3 =  torch.Size([5, 128, 8, 160, 160])
#         x3 = self.Maxpool(x2)
# #         print("4 = " ,x3.shape) 4 =  torch.Size([5, 128, 4, 80, 80])
#         x3 = self.Conv3(x3)
# #         print("5 = " ,x3.shape) 5 =  torch.Size([5, 256, 4, 80, 80])

#         x4 = self.Maxpool(x3)
# #         print("6 = " ,x4.shape) 6 =  torch.Size([5, 256, 2, 40, 40])
#         x4 = self.Conv4(x4)
# #         print("7 = " ,x4.shape) 7 =  torch.Size([5, 512, 2, 40, 40])

#         x5 = self.Maxpool(x4)
# #         print("8 = " ,x5.shape) 8 =  torch.Size([5, 512, 1, 20, 20])
#         x5 = self.Conv5(x5)
#         # print("9 = ", x5.shape) # 9 =  torch.Size([5, 1024, 1, 20, 20])
        
#         # Aux 붙이기
#         cls = self.linear(x5.squeeze())
# #         print("10 = " ,cls.shape)  10 =  torch.Size([5, 1])

#         # decoding + concat path
#         d5 = self.Up5(x5)
# #         print("11 = " ,d5.shape) 11 =  torch.Size([5, 512, 2, 40, 40])
#         d5 = torch.cat((x4,d5),dim=1)
# #         print("12 = " ,d5.shape)  torch.Size([5, 1024, 2, 40, 40])
#         d5 = self.Up_conv5(d5)
# #         print("13 = " ,d5.shape) torch.Size([5, 512, 2, 40, 40])
        
#         d4 = self.Up4(d5)
# #         print("14 = " ,d4.shape)  torch.Size([5, 256, 4, 80, 80])
#         d4 = torch.cat((x3,d4),dim=1)
# #         print("15 = " ,d4.shape)   torch.Size([5, 512, 4, 80, 80])
#         d4 = self.Up_conv4(d4)
# #         print("16 = " ,d4.shape) torch.Size([5, 256, 4, 80, 80])

#         d3 = self.Up3(d4)
# #         print("17 = " ,d3.shape)   torch.Size([5, 128, 8, 160, 160])
#         d3 = torch.cat((x2,d3),dim=1)
# #         print("18 = " ,d3.shape)     torch.Size([5, 256, 8, 160, 160])
#         d3 = self.Up_conv3(d3)
# #         print("19 = " ,d3.shape)    torch.Size([5, 128, 8, 160, 160])

#         d2 = self.Up2(d3)
# #         print("20 = " ,d2.shape)     torch.Size([5, 64, 16, 320, 320])
#         d2 = torch.cat((x1,d2),dim=1)
# #         print("21 = " ,d2.shape)      torch.Size([5, 128, 16, 320, 320])
#         d2 = self.Up_conv2(d2)
# #         print("22 = " ,d2.shape)      torch.Size([5, 64, 16, 320, 320])

#         d1 = self.Conv_1x1(d2)
# #         print("1 = " ,d1.shape)       torch.Size([5, 1, 16, 320, 320])

#         return d1, cls


########################################################################################################################
##################### 모델 2##########################
########################################################################################################################
class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block,self).__init__()
        self.timedistributed1 = TimeDistributed(module=nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, padding=1, bias=True), batch_first=True)
        self.norm1 = nn.BatchNorm3d(ch_out)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.timedistributed2 = TimeDistributed(module=nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, stride=1, padding=1, bias=True), batch_first=True)
        self.norm2 = nn.BatchNorm3d(ch_out)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self,x):
        x = self.timedistributed1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.timedistributed2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        return x
    
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.convlstm = ConvLSTM2d(input_dim=ch_in, hidden_dim=ch_out, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True, return_all_layers=False)
        self.norm = nn.BatchNorm3d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        
        
    def forward(self,x):
        x = self.up(x)
#         print("check1",x.shape) check1 torch.Size([5, 1024, 2, 40, 40])
        x = self.convlstm(x)
#         print("check2",x.shape) torch.Size([5, 512, 2, 40, 40])
        x = self.norm(x[0])
        x = self.relu(x)
        return x
    
    
class U_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv3d(64, output_ch, kernel_size=1, stride=1, padding=0)
        
        self.linear = nn.Sequential(nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(1024, 1, bias=True))

    def forward(self, x):
        # encoding path
#         print("x = " ,x.shape) torch.Size([5, 1, 16, 320, 320])
        x1 = self.Conv1(x)
#         print("1 = " ,x1.shape) 1 =  torch.Size([5, 64, 16, 320, 320])
        x2 = self.Maxpool(x1)
#         print("2 = " ,x2.shape) 2 =  torch.Size([5, 64, 8, 160, 160])
        x2 = self.Conv2(x2)
#         print("3 = " ,x2.shape) 3 =  torch.Size([5, 128, 8, 160, 160])
        x3 = self.Maxpool(x2)
#         print("4 = " ,x3.shape) 4 =  torch.Size([5, 128, 4, 80, 80])
        x3 = self.Conv3(x3)
#         print("5 = " ,x3.shape) 5 =  torch.Size([5, 256, 4, 80, 80])

        x4 = self.Maxpool(x3)
#         print("6 = " ,x4.shape) 6 =  torch.Size([5, 256, 2, 40, 40])
        x4 = self.Conv4(x4)
#         print("7 = " ,x4.shape) 7 =  torch.Size([5, 512, 2, 40, 40])

        x5 = self.Maxpool(x4)
#         print("8 = " ,x5.shape) 8 =  torch.Size([5, 512, 1, 20, 20])
        x5 = self.Conv5(x5)
        # print("9 = ", x5.shape) # 9 =  torch.Size([5, 1024, 1, 20, 20])
        
        # Aux 붙이기
        cls = self.linear(x5.squeeze())
#         print("10 = " ,cls.shape)  10 =  torch.Size([5, 1])

        # decoding + concat path
        d5 = self.Up5(x5)
#         print("11 = " ,d5.shape) 11 =  torch.Size([5, 512, 2, 40, 40])
        d5 = torch.cat((x4,d5),dim=1)
#         print("12 = " ,d5.shape)  torch.Size([5, 1024, 2, 40, 40])
        d5 = self.Up_conv5(d5)
#         print("13 = " ,d5.shape) torch.Size([5, 512, 2, 40, 40])
        
        d4 = self.Up4(d5)
#         print("14 = " ,d4.shape)  torch.Size([5, 256, 4, 80, 80])
        d4 = torch.cat((x3,d4),dim=1)
#         print("15 = " ,d4.shape)   torch.Size([5, 512, 4, 80, 80])
        d4 = self.Up_conv4(d4)
#         print("16 = " ,d4.shape) torch.Size([5, 256, 4, 80, 80])

        d3 = self.Up3(d4)
#         print("17 = " ,d3.shape)   torch.Size([5, 128, 8, 160, 160])
        d3 = torch.cat((x2,d3),dim=1)
#         print("18 = " ,d3.shape)     torch.Size([5, 256, 8, 160, 160])
        d3 = self.Up_conv3(d3)
#         print("19 = " ,d3.shape)    torch.Size([5, 128, 8, 160, 160])

        d2 = self.Up2(d3)
#         print("20 = " ,d2.shape)     torch.Size([5, 64, 16, 320, 320])
        d2 = torch.cat((x1,d2),dim=1)
#         print("21 = " ,d2.shape)      torch.Size([5, 128, 16, 320, 320])
        d2 = self.Up_conv2(d2)
#         print("22 = " ,d2.shape)      torch.Size([5, 64, 16, 320, 320])

        d1 = self.Conv_1x1(d2)
#         print("1 = " ,d1.shape)       torch.Size([5, 1, 16, 320, 320])

        return d1, cls




