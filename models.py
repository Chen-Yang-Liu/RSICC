import torch
from torch import nn
import torchvision
from torchvision.transforms import Resize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda:0


class CNN_Encoder(nn.Module):
    """
    CNN_Encoder.
    """

    def __init__(self, NetType, method, encoded_image_size=14):
        super(CNN_Encoder, self).__init__()
        self.NetType = NetType
        self.enc_image_size = encoded_image_size

        if 'resnet' in NetType:
            # raise ValueError('Feature extraction only supports ResNets')
            cnn = getattr(torchvision.models,NetType)(pretrained=True)
            layers = [
                cnn.conv1,
                cnn.bn1,
                cnn.relu,
                cnn.maxpool,
            ]

            # 使用前model_stage个卷积块（最大为4）提取特征
            model_stage = 3
            for i in range(model_stage):
                name = 'layer%d' % (i + 1)
                layers.append(getattr(cnn, name))
            self.net = nn.Sequential(*layers)
        if 'vgg' in NetType:
            net = torchvision.models.vgg16(pretrained=True)
            modules = list(net.children())[:-1]
            self.net = nn.Sequential(*modules)
        if 'vit' in NetType:  # "vit_b_16"
            net = getattr(torchvision.models,NetType)(pretrained=True)
            self.net = net

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images [batch_size, encoded_image_size=14, encoded_image_size=14, 2048]
        """
        self.NetType = 'resnet'
        if 'resnet' in self.NetType:
            out = self.net(images)  # (batch_size, 2048, image_size/32, image_size/32)
            out = self.adaptive_pool(out)  # [batch_size, 2048/512, 8, 8] -> [batch_size, 2048/512, 14, 14]
            # out = out.permute(0, 2, 3, 1)
        if 'vgg' in self.NetType:
            out = self.net(images)  # (batch_size, 2048, image_size/32, image_size/32)
            out = self.adaptive_pool(out)  # [batch_size, 2048/512, 8, 8] -> [batch_size, 2048/512, 14, 14]
            # out = out.permute(0, 2, 3, 1)
        if 'vit' in self.NetType:
            torch_resize = Resize([224, 224])  # 定义Resize类对象
            images = torch_resize(images)

            # images = self.adaptive_pool2(images)
            x = self.net._process_input(images)
            n = x.shape[0]
            # Expand the class token to the full batch
            batch_class_token = self.net.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.net.encoder(x)
            # Classifier "token" as used by standard language architectures
            x = x[:, 1:,:]
            # x = self.net.heads(x)

            # out = self.net(images)  # (batch_size, 2048, image_size/32, image_size/32)
            # dif=out-x
            out =x
            out = out.permute(0,2,1).view(n, -1, 14,14)

        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.net.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.net.children())[5:]:  #
            for p in c.parameters():
                p.requires_grad = fine_tune