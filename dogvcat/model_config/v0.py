from torch import nn, optim
import torchvision
from torchvision.models import resnet18

weights = torchvision.models.ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

model.fc = nn.Identity()

# freezing the model
for param in model.parameters():
    param.requires_grad = False