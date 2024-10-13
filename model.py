import torch



# class ResBlock(torch.nn.Module):
#     def __init__(self,in_channels, out_channels, stride):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.stride = stride
#
#
#         self.conv1 =  torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride = self.stride, padding=1, bias=False)
#         self.batch_norm1 = torch.nn.BatchNorm2d(self.out_channels)
#         self.relu = torch.nn.ReLU()
#
#         self.conv2 = torch.nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.batch_norm2 = torch.nn.BatchNorm2d(self.out_channels)
#
#         self.downsample = torch.nn.Sequential(
#             torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=stride, padding=0,  bias=False),
#             torch.nn.BatchNorm2d(self.out_channels)
#         )
#
#     def forward(self, input_tensor):
#         residual = input_tensor
#         output_tensor = self.conv1(input_tensor)
#         #print("conv1",output_tensor.shape)
#         output_tensor = self.batch_norm1(output_tensor)
#         #print("bn1", output_tensor.shape)
#         output_tensor = self.relu(output_tensor)
#         #print("relu1", output_tensor.shape)
#         output_tensor = self.conv2(output_tensor)
#         #print("conv2", output_tensor.shape)
#         output_tensor = self.batch_norm2(output_tensor)
#
#         #print(output_tensor.shape)
#         #print(residual.shape)
#
#         if residual.shape != output_tensor.shape:
#             residual = self.downsample(residual)
#             #print("downsample", residual.shape)
#
#         output_tensor += residual
#         output_tensor = self.relu(output_tensor)
#         return output_tensor
#
#
# class ResNet(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.conv = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False)
#         self.batch_norm = torch.nn.BatchNorm2d(64)
#         self.relu = torch.nn.ReLU(inplace=True)
#         self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2)
#
#         self.resblock1 = ResBlock(64, 64, stride=1)
#         self.resblock2 = ResBlock(64, 128, stride=2)
#         self.resblock3 = ResBlock(128, 256, stride=2)
#         self.resblock4 = ResBlock(256, 512, stride=2)
#
#         self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
#         #self.avgpool = torch.nn.AvgPool2d(kernel_size =10)
#         self.flatten = torch.nn.Flatten()
#         self.fc = torch.nn.Linear(512, 2)
#         self.sigmoid = torch.nn.Sigmoid()
#
#         self.dropout = torch.nn.Dropout(p=0.5)
#
#     def forward(self, input_img):
#         output_tensor = self.conv(input_img)
#         output_tensor = self.batch_norm(output_tensor)
#         output_tensor = self.relu(output_tensor)
#         output_tensor = self.maxpool(output_tensor)
#
#         output_tensor = self.resblock1(output_tensor)
#         output_tensor = self.resblock2(output_tensor)
#         output_tensor = self.resblock3(output_tensor)
#         output_tensor = self.resblock4(output_tensor)
#
#         #output_tensor = self.dropout(output_tensor)
#
#         output_tensor = self.avgpool(output_tensor)
#         output_tensor = self.flatten(output_tensor)
#         output_tensor = self.fc(output_tensor)
#         output_tensor = self.sigmoid(output_tensor)
#         return output_tensor
#


"""
import torch

class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm1 = torch.nn.BatchNorm2d(self.out_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.batch_norm2 = torch.nn.BatchNorm2d(self.out_channels)
        self.conv3 = torch.nn.Conv2d(self.out_channels, 4*self.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm3 = torch.nn.BatchNorm2d(4*self.out_channels)

        self.downsample = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels, 4*self.out_channels, kernel_size=1, stride=self.stride, padding=0, bias=False),
            torch.nn.BatchNorm2d(4*self.out_channels)
        )

    def forward(self, input_tensor):
        residual = input_tensor
        output_tensor = self.conv1(input_tensor)
        output_tensor = self.batch_norm1(output_tensor)
        output_tensor = self.relu(output_tensor)
        output_tensor = self.conv2(output_tensor)
        output_tensor = self.batch_norm2(output_tensor)
        output_tensor = self.relu(output_tensor)
        output_tensor = self.conv3(output_tensor)
        output_tensor = self.batch_norm3(output_tensor)

        if residual.shape != output_tensor.shape:
            residual = self.downsample(residual)

        output_tensor += residual
        output_tensor = self.relu(output_tensor)
        return output_tensor

class ResNet(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.resblock1 = self._make_layer(64, 3, stride=1)
        self.resblock2 = self._make_layer(128, 4, stride=2)
        self.resblock3 = self._make_layer(256, 6, stride=2)
        self.resblock4 = self._make_layer(512, 3, stride=2)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512, num_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResBlock(64, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels*4, out_channels, stride=1))
        return torch.nn.Sequential(*layers)

    def forward(self, input_tensor):
        output_tensor = self.conv1(input_tensor)
        output_tensor = self.batch_norm1(output_tensor)
        output_tensor = self.relu(output_tensor)
        output_tensor = self.maxpool(output_tensor)

        output_tensor = self.resblock1(output_tensor)
        output_tensor = self.resblock2(output_tensor)
        output_tensor = self.resblock3(output_tensor)
        output_tensor = self.resblock4(output_tensor)

        output_tensor = self.avgpool(output_tensor)
        output_tensor = torch.flatten(output_tensor, 1)
        output_tensor = self.fc(output_tensor)
        output_tensor = self.sigmoid(output_tensor)
        return output_tensor
"""

import torch
import torchvision.models as models

import torch
import torchvision.models as models

class ResNet(torch.nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        resnet = models.resnet50(weights=pretrained)
        self.features = torch.nn.Sequential(*list(resnet.children())[:-2])  # Extract layers up to avgpool
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = torch.nn.Linear(2048, 512)  # Adjust the number of output units
        self.bn1= torch.nn.BatchNorm1d(512)
        self.fc2 = torch.nn.Linear(512, 256)  # Adjust the number of output units
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.fc3 = torch.nn.Linear(256, num_classes)  # Adjust the number of output units
        self.sigmoid = torch.nn.Sigmoid()

        # Freeze layers except the last three
        if pretrained:
            self._freeze_layers(self.features, freeze=True)
            self._freeze_layers(self.features[-3:], freeze=False)

    def _freeze_layers(self, model, freeze=True):
        for param in model.parameters():
            param.requires_grad = not freeze

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


