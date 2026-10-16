import torch.nn as nn
import timm
from models.activations import get_activation

class CustomCoaT(nn.Module):
    def __init__(self, activation_name="tanhexp", num_classes=3, pretrained=True, dropout=0.4):
        super().__init__()
        self.backbone = timm.create_model("coat_lite_medium", pretrained=pretrained, num_classes=0, img_size=224)
        in_feats = getattr(self.backbone, "num_features", None)
        if in_feats is None:
            raise RuntimeError("No se pudo inferir 'num_features' del backbone.")

        act_layer = get_activation(activation_name)
        self.classifier = nn.Sequential(
            nn.Linear(in_feats, 512),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        out = self.classifier(feat)
        return out

