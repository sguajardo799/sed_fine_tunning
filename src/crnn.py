import torch
import torch.nn as nn
import torchaudio
from torchvision.models import resnet18, ResNet18_Weights

class CRNN(nn.Module):
    def __init__(self, num_classes=7, sr=32000, n_fft=1024, hop_length=320, n_mels=64, pretrained=True):
        super().__init__()
        
        # Log-Mel Spectrogram Extractor
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        self.num_classes = num_classes
        
        # CNN (ResNet18)
        # We use a pretrained ResNet18 but modify the first layer for 1 channel input
        # and remove the pooling/fc layers.
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = resnet18(weights=weights)
        
        # Modify first conv layer to accept 1 channel (spectrogram) instead of 3 (RGB)
        # We can sum the weights of the 3 channels to initialize or just reset.
        # Summing is a common trick to keep pretrained weights useful.
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.resnet.conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)
            
        # Remove the fully connected layer and average pooling
        # We want to keep the time dimension.
        # ResNet18 structure: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4
        # layer4 output: [Batch, 512, F_dim, T_dim]
        # We will pool over frequency dimension and keep time.
        self.cnn_layers = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        )
        
        # RNN (BiGRU)
        # Input dim to RNN depends on ResNet output.
        # ResNet18 layer4 output channels is 512.
        # We will pool frequency dimension, so input to RNN is 512.
        self.rnn = nn.GRU(input_size=512, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        
        # Classifier
        # BiGRU output is 2 * hidden_size = 512
        self.head = nn.Linear(512, num_classes)
        
        self.sr = sr
        self.hop_length = hop_length

    def forward(self, x):
        # x: [Batch, Samples]
        
        # Extract Log-Mel Spectrogram
        x = self.mel_spec(x) # [Batch, n_mels, Time]
        x = self.amplitude_to_db(x)
        
        # Add channel dimension for CNN
        x = x.unsqueeze(1) # [Batch, 1, n_mels, Time]
        
        # CNN
        x = self.cnn_layers(x) # [Batch, 512, F_dim, T_dim]
        
        # Pool over frequency dimension
        # We can use max pool or avg pool. Let's use avg pool.
        x = x.mean(dim=2) # [Batch, 512, T_dim]
        
        # Prepare for RNN
        x = x.permute(0, 2, 1) # [Batch, T_dim, 512]
        
        # RNN
        x, _ = self.rnn(x) # [Batch, T_dim, 512]
        
        # Classifier
        logits = self.head(x) # [Batch, T_dim, Num_Classes]
        
        return logits

    def get_time_resolution(self):
        # Calculate time resolution based on hop_length and CNN strides
        # MelSpec hop: 320 samples
        # ResNet18 strides:
        # conv1: stride 2
        # maxpool: stride 2
        # layer1: stride 1
        # layer2: stride 2
        # layer3: stride 2
        # layer4: stride 2
        # Total CNN stride: 2 * 2 * 1 * 2 * 2 * 2 = 32
        
        # Total stride in samples = hop_length * cnn_stride = 320 * 32 = 10240
        # Time resolution = 10240 / sr = 10240 / 32000 = 0.32s
        
        total_stride = 32
        frame_samples = self.hop_length * total_stride
        return frame_samples / self.sr
