import torch.nn as nn
import torchvision.models as models

class ResnetEncoder(nn.Module):
    def __init__(self, num_layers, pretrained, num_input_images):
        super(ResnetEncoder, self).__init__()
        self.encoder = models.resnet50(pretrained=pretrained)
        self.num_ch_enc = self.encoder.fc.in_features
        del self.encoder.fc

    def forward(self, input_image):
        x = self.encoder.conv1(input_image)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        return x
