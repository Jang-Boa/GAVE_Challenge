import torch
import torch.nn as nn
<<<<<<< HEAD
from utils import initialize_weights
=======
from .utils import initialize_weights
>>>>>>> d4b3098 (update)


class UNet(nn.Module):
    def __init__(self, encoder, num_classes=1, num_features=[128, 256, 512, 1024]):
        super(UNet, self).__init__()
        model = encoder # load_pretrained(weight_path)
        self.conv1 = list(model.encoder.children())[0] # nn.Sequential(*list(model.module.children())[:1])
        self.bn1 = list(model.encoder.children())[1]
        self.relu = list(model.encoder.children())[2]
        self.maxpool = list(model.encoder.children())[3]
        self.layer1 = list(model.encoder.children())[4]
        self.layer2 = list(model.encoder.children())[5]
        self.layer3 = list(model.encoder.children())[6]
        self.layer4 = list(model.encoder.children())[7]
        
        self.up4 = self.__upsample__(in_feature=num_features[3]*2, out_feature=num_features[2])
        for m in self.up4.modules():
            m.apply(initialize_weights)
            
        self.up3 = self.__upsample__(in_feature=num_features[2]*3, out_feature=num_features[1])
        for m in self.up3.modules():
            m.apply(initialize_weights)
            
        self.up2 = self.__upsample__(in_feature=num_features[1]*3, out_feature=num_features[0])
        for m in self.up2.modules():
            m.apply(initialize_weights)
            
        self.up1 = self.__upsample__(in_feature=num_features[0]*3, out_feature=64)
        for m in self.up1.modules():
            m.apply(initialize_weights)
        
        self.final_conv = nn.ConvTranspose2d(in_channels=64, out_channels=num_classes, kernel_size=2, stride=2)
        
        
    def __upsample__(self, in_feature, out_feature):
        layers = []
        
        mid_feature = int(in_feature/2)
        for layer in range(2):
            layers += [nn.Conv2d(in_channels=in_feature, out_channels=mid_feature, kernel_size=3, stride=1, padding=1), 
                       nn.BatchNorm2d(num_features=mid_feature), 
                       nn.ReLU()]
            in_feature = mid_feature
        layers += [nn.ConvTranspose2d(in_channels=mid_feature, out_channels=out_feature, kernel_size=2, stride=2)]
            
        return nn.Sequential(*layers)        
    
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        u4 = self.up4(x4)
        u4 = torch.cat((u4, x3), dim=1)
        
        u3 = self.up3(u4)
        u3 = torch.cat((u3, x2), dim=1)
        
        u2 = self.up2(u3)
        u2 = torch.cat((u2, x1), dim=1)
        
        u1 = self.up1(u2)
        
        output = self.final_conv(u1)
        output = torch.sigmoid(output)
        return output
