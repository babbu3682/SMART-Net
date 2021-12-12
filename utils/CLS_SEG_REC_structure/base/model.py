import torch
from . import initialization as init


class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_decoder(self.recon_decoder)
        init.initialize_head(self.segmentation_head)
        init.initialize_head(self.classification_head)
        init.initialize_head(self.recon_head)

    def forward(self, x0):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features0  = self.encoder(x0)

        # Seg
        decoder_output = self.decoder(*features0)
        masks = self.segmentation_head(decoder_output)

        # Cls
        labels = self.classification_head(features0[-1])
        
        # Recon
        recon_decoder_output = self.recon_decoder(features0[-1])
        recon = self.recon_head(recon_decoder_output)

        return masks, labels, recon



    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x
