import torch.nn.functional as F
import torch.nn as nn
import torch

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels  = self.input_dim + self.hidden_dim,
                              out_channels = 4*self.hidden_dim, # 4를 곱하고 4개로 찢음
                              kernel_size  = self.kernel_size,
                              padding      = self.padding,
                              bias         = self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # h_cur 이전의 값, input_tensor 지금의 값
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        # gates
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)

        # Cell 생성 gate
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),   # B, C, H, W 형태의 zero
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

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
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        # 이놈이 복병 ㅎㅎㅎ;;;;
        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
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

#############################################################################################################
# Custom CRNN

class Real_ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, stride, padding, cnn_dropout, rnn_dropout, bias=True):
        super(Real_ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        self.input_conv = nn.Conv2d(in_channels = self.input_dim, 
                                out_channels = 4*self.hidden_dim,
                                kernel_size = self.kernel_size,
                                stride = self.stride,
                                padding = self.padding,
                                bias = self.bias)

        self.rnn_conv = nn.Conv2d(in_channels = self.hidden_dim, 
                                out_channels = 4*self.hidden_dim, 
                                kernel_size = self.kernel_size,
                                stride = 1, # reference tensorflow
                                padding = self.padding)
            
        self.cnn_dropout = nn.Dropout(cnn_dropout, inplace=False)
        self.rnn_dropout = nn.Dropout(rnn_dropout, inplace=False)
        
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # CNN
        x = self.cnn_dropout(input_tensor)
        x_conv = self.input_conv(x)
        # separate i, f, c, o
        x_i, x_f, x_c, x_o = torch.split(x_conv, self.hidden_dim, dim=1)
        
        # LSTM
        h = self.rnn_dropout(h_cur)
        h_conv = self.rnn_conv(h)
        
        # separate i, f, c, o
        h_i, h_f, h_c, h_o = torch.split(h_conv, self.hidden_dim, dim=1)
        
        # Gate
        f = torch.sigmoid((x_f + h_f))
        i = torch.sigmoid((x_i + h_i))
        o = torch.sigmoid(x_o + h_o)

        c_next = f * c_cur + i * torch.tanh(x_c + h_c)
        h_next = o * torch.tanh(c_next)
        
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.input_conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.input_conv.weight.device))


class Real_ConvLSTM(nn.Module):
    def __init__(self, 
                input_dim, 
                hidden_dim, 
                kernel_size, 
                stride, 
                padding, 
                bias=False,
                cnn_dropout=0.2, 
                rnn_dropout=0.2, 
                bidirectional=False):
        super(Real_ConvLSTM, self).__init__()

        # self.bidirectional = bidirectional
        self.bidirectional = False
        self.LSTM_cell = Real_ConvLSTMCell(input_dim = input_dim,
                                        hidden_dim = hidden_dim,
                                        kernel_size = kernel_size,
                                        stride = stride,
                                        padding = padding,
                                        cnn_dropout = cnn_dropout,
                                        rnn_dropout = rnn_dropout,
                                        bias = bias)
        if (self.bidirectional==True):
            self.inverse_LSTM_cell = Real_ConvLSTMCell(input_dim = input_dim,
                                                    hidden_dim = hidden_dim,
                                                    kernel_size = kernel_size,
                                                    stride = stride,
                                                    padding = padding,
                                                    cnn_dropout = cnn_dropout,
                                                    rnn_dropout = rnn_dropout,
                                                    bias = bias)

    
    def forward(self, input_tensor, hidden_state=None):
        # input_tensor shape = (B, C, D, H, W) 
        b, _, seq_len, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError() # "아직 구현하지 않은 부분입니다"
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
            
            if self.bidirectional is True:
                hidden_state_inverse = self._init_hidden(batch_size=b, image_size=(h, w))

        ## LSTM forward direction
        input_fw = input_tensor
        h, c = hidden_state

        # if stride 2
        if (self.LSTM_cell.input_conv.stride[0] == 2):
            h = F.max_pool3d(input=h, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
            c = F.max_pool3d(input=c, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

        output_inner = []
        for t in range(seq_len):
            h, c = self.LSTM_cell(input_tensor=input_fw[:, :, t, :, :], cur_state=[h, c])
            output_inner.append(h)

        output_inner = torch.stack(output_inner, dim=2)
        
        layer_output = output_inner
        last_state = [h, c]
        
        ## LSTM inverse direction
        if self.bidirectional is True:
            input_inverse = input_tensor.flip(dims=[2])
            h_inverse, c_inverse = hidden_state_inverse

            # if stride 2
            if (self.inverse_LSTM_cell.input_conv.stride[0] == 2):
                h_inverse = F.max_pool3d(input=h_inverse, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
                c_inverse = F.max_pool3d(input=c_inverse, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

            output_inverse = []
            for t in range(seq_len):
                h_inverse, c_inverse = self.inverse_LSTM_cell(input_tensor=input_inverse[:, :, t, :, :], cur_state=[h_inverse, c_inverse])
                output_inverse.append(h_inverse)

            output_inverse.reverse() # 뒤집혀 있기에 다시 뒤집어야 한다..
            output_inverse = torch.stack(output_inverse, dim=2)
            
            layer_output = torch.cat((output_inner, output_inverse), dim=1)
            last_state_inverse = [h_inverse, c_inverse]
        
        # return layer_output, last_state, last_state_inverse if self.bidirectional is True else None
        return layer_output

    def _init_hidden(self, batch_size, image_size):
        init_states = self.LSTM_cell.init_hidden(batch_size, image_size)
        return init_states



################################################################# BD LSTM #########################################################################
# inverse 사용 할꺼면 input 뒤집고 output도 다시 한번 뒤집어서 Concat 해야 한다!!

class BD_ConvLSTM(nn.Module):
    # Constructor
    def __init__(self, input_channels=16, hidden_channels=16, kernel_size=3, stride=1, padding=1, bias=False, cnn_dropout=0.2, rnn_dropout=0.2, num_classes=2, bidirectional=True):
        super(BD_ConvLSTM, self).__init__()
        self.BD_convlstm_1 = Real_ConvLSTM(input_dim=input_channels, hidden_dim=hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, 
                                        cnn_dropout=cnn_dropout, rnn_dropout=rnn_dropout, bidirectional=bidirectional)
        self.bn_1 = nn.InstanceNorm3d(num_features=hidden_channels)

        self.BD_convlstm_2 = Real_ConvLSTM(input_dim=2*hidden_channels, hidden_dim=hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, 
                                        cnn_dropout=cnn_dropout, rnn_dropout=rnn_dropout, bidirectional=bidirectional)
        self.bn_2 = nn.InstanceNorm3d(num_features=hidden_channels)
      
        # self.reverse_net = ConvLSTM(input_dim=input_channels, hidden_dim=hidden_channels, kernel_size=(kernel_size, kernel_size), num_layers=num_layers, bias=bias, batch_first=True, return_all_layers=False)
        self.conv = nn.Conv3d( 2*hidden_channels, num_classes, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1) )

    
    def forward(self, x):
        y_out = self.BD_convlstm_1(x)[0]
        y_out = self.bn_1(y_out)

        y_out = self.BD_convlstm_2(y_out)[0]
        y_out = self.bn_2(y_out)

        y_out = self.conv(y_out)

        return y_out


class BD_LSTM(nn.Module):
    # Constructor
    def __init__(self, input_channels=384, hidden_channels=384, num_classes=1, bias=True, num_layers=2, dropout_p=0.4, bidirectional=True):
        super(BD_LSTM, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(output_size=(None, 1, 1)) # nn.AdaptiveMaxPool3d(output_size=(None, 1, 1))
        
        self.BD_lstm = nn.LSTM(input_size=input_channels, hidden_size=hidden_channels, num_layers=num_layers, bias=bias, batch_first=True, dropout=dropout_p, bidirectional=bidirectional)
        # self.reverse_net = nn.LSTM(input_size=input_channels, hidden_size=hidden_channels, num_layers=num_layers, bias=bias, batch_first=True) # input 형태 (batch, seq, feature)

        self.fc1 = nn.Linear( 2*hidden_channels, hidden_channels, bias=True )
        # self.bn1 = nn.BatchNorm1d(hidden_channels, momentum=0.01)
        # self.bn1 = nn.InstanceNorm1d(hidden_channels, momentum=0.01)
        self.bn1 = nn.GroupNorm(num_groups=32, num_channels=hidden_channels)
        self.relu1 = nn.ReLU(inplace=False)

        self.fc2 = nn.Linear( hidden_channels, hidden_channels//2, bias=True )
        # self.bn2 = nn.BatchNorm1d(hidden_channels//2, momentum=0.01)
        # self.bn2 = nn.InstanceNorm1d(hidden_channels//2, momentum=0.01)
        self.bn2 = nn.GroupNorm(num_groups=32, num_channels=hidden_channels//2)
        self.relu2 = nn.ReLU(inplace=False)

        self.dropout = nn.Dropout(p=dropout_p, inplace=False)
        self.linear = nn.Linear( hidden_channels//2, num_classes, bias=True )
        self.activation = nn.Sigmoid()


    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.shape[0], x.shape[1], -1) # (B, C, D)

        # Bottle neck input은 (B, C, D) ---> LSTM input 형태 (B, Seq, Feature) 로 바꿔야 함
        x = x.permute(0, 2, 1).contiguous()  

        self.BD_lstm.flatten_parameters()  # nn.DataParallel
        # self.reverse_net.flatten_parameters()  # nn.DataParallel

        y_cat = self.BD_lstm(x)[0]
        # y_reverse_out = self.reverse_net(x.flip(dims=[1]))[0]

        # y_cat = torch.cat((y_forward_out[:, -1, :], y_reverse_out[:, -1, :]), dim=1)

        y = self.relu1(self.bn1(self.fc1(y_cat[:, -1, :])))
        y = self.relu2(self.bn2(self.fc2(y)))
        y = self.dropout(y)
        y = self.linear(y)
        y = self.activation(y)
        
        return y