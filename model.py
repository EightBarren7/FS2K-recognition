import torch
from torch import nn
import torchvision.models as models
from convnext import convnext_base, convnext_xlarge


class Identity(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 2)
        self.fc2 = nn.Linear(input_size, 5)
        self.fc3 = nn.Linear(input_size, 2)
        self.fc4 = nn.Linear(input_size, 2)
        self.fc5 = nn.Linear(input_size, 2)
        self.fc6 = nn.Linear(input_size, 2)

    def forward(self, feature):
        label1 = self.fc1(feature)
        label2 = self.fc2(feature)
        label3 = self.fc3(feature)
        label4 = self.fc4(feature)
        label5 = self.fc5(feature)
        label6 = self.fc6(feature)
        output = [label1, label2, label3, label4, label5, label6]
        return output


class IdentityV2(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 2)
        self.fc2 = nn.Linear(input_size, 2)
        self.fc3 = nn.Linear(input_size, 2)
        self.fc4 = nn.Linear(input_size, 2)
        self.fc5 = nn.Linear(input_size, 2)

    def forward(self, feature):
        label1 = self.fc1(feature)
        label2 = self.fc2(feature)
        label3 = self.fc3(feature)
        label4 = self.fc4(feature)
        label5 = self.fc5(feature)
        output = [label1, label2, label3, label4, label5]
        return output


def get_model(model_name, model_version):
    model = None
    if model_name == 'ConvNeXt':
        model = convnext_base(10)
        weights_dict = torch.load('./result/convnext_base_22k_224.pth')['model']
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        model.load_state_dict(weights_dict, strict=False)
        if model_version == 'v1':
            model.head = Identity(1024)
        if model_version == 'v2':
            model.head = IdentityV2(1024)
    elif model_name == 'ConvNeXtv2':
        model = convnext_xlarge(10)
        weights_dict = torch.load('./result/convnext_xlarge_22k_224.pth')['model']
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        model.load_state_dict(weights_dict, strict=False)
        if model_version == 'v1':
            model.head = Identity(2048)
        if model_version == 'v2':
            model.head = IdentityV2(2048)
    else:
        if model_name == 'ResNet50':
            model = models.resnet50(pretrained=True)
        if model_name == 'ResNet50v2':
            model = models.resnet50(pretrained=False)
        if model_name == 'ResNet152':
            model = models.resnet152(pretrained=True)
        if model_name == 'ResNet152v2':
            model = models.resnet152(pretrained=False)
        if model_version == 'v1':
            model.fc = Identity(2048)
        elif model_version == 'v2':
            model.fc = IdentityV2(2048)
    return model
