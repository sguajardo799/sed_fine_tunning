import torch
import torch.nn as nn
from hear21passt.base import get_basic_model

class FineTunePaSST(nn.Module):
    def __init__(self, num_classes=7, pretrained=True, freeze_encoder=False):
        super().__init__()
        # Load pre-trained PaSST using the wrapper to handle MelSpectrogram
        self.passt = get_basic_model(mode="logits")
        
        # We need the sequence output, but PaSST pools it.
        # We will use a forward hook to capture the output of the normalization layer (before pooling).
        self.features = None
        def hook(module, input, output):
            self.features = output
            
        # Register hook on the norm layer of the internal PaSST model
        self.passt.net.norm.register_forward_hook(hook)

        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.passt.parameters():
                param.requires_grad = False
                
        # New classification head for SED (frame-wise)
        # PaSST embedding dim is 768
        self.embed_dim = 768
        self.num_classes = num_classes
        
        self.head = nn.Linear(self.embed_dim, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [Batch, Samples]
        
        # Run the model to trigger the hook
        # We don't care about the output logits
        _ = self.passt(x)
        
        # Get captured features
        x = self.features # [Batch, Seq_Len, Dim]
        self.features = None # Clear
        
        # Remove CLS and Distillation tokens
        # Usually CLS is at 0, Dist at 1 (if present)
        # PaSST usually has 2 tokens if distillation is used.
        # Let's check the sequence length.
        # Log showed: 1190 total. 1188 patches. So 2 tokens.
        # We skip the first 2.
        x = x[:, 2:, :]
        
        x = self.dropout(x)
        logits = self.head(x) # [Batch, Num_Patches, Num_Classes]
        
        return logits

    def get_time_resolution(self, input_duration=10.0):
        # 10s -> ~1000 frames -> 99 time patches (from log)
        # 10s / 99 = ~0.1s
        return 10.0 / 99.0

