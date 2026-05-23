import timm
import torch.nn as nn

class DeepFakeModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=1
        )

    def forward(self, x):

        return self.model(x)