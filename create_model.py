from arch.smart_net import *




# Create Model
def create_model(stream, name):
    if stream == 'Upstream':
        
        if name == 'Up_SMART_Net':
            model = Up_SMART_Net()     

        ## Dual    
        elif name == 'Up_SMART_Net_Dual_CLS_SEG':
            model = Up_SMART_Net_Dual_CLS_SEG()

        elif name == 'Up_SMART_Net_Dual_CLS_REC':
            model = Up_SMART_Net_Dual_CLS_REC()

        elif name == 'Up_SMART_Net_Dual_SEG_REC':
            model = Up_SMART_Net_Dual_SEG_REC()

        ## Single
        elif name == 'Up_SMART_Net_Single_CLS':
            model = Up_SMART_Net_Single_CLS()

        elif name == 'Up_SMART_Net_Single_SEG':
            model = Up_SMART_Net_Single_SEG()

        elif name == 'Up_SMART_Net_Single_REC':
            model = Up_SMART_Net_Single_REC()
        else :
            raise KeyError("Wrong model name `{}`".format(name))        

    elif stream == 'Downstream':

        if name == 'Down_SMART_Net_CLS':
            model = Down_SMART_Net_CLS()

        elif name == 'Down_SMART_Net_SEG':
            model = Down_SMART_Net_SEG()

        else :
            raise KeyError("Wrong model name `{}`".format(name))               


    else :
        raise KeyError("Wrong stream name `{}`".format(stream))
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Learnable Params:', n_parameters)   

    return model