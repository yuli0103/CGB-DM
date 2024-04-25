import torch
import torch.nn as nn

# --------------------------------#
# 从torch官方可以下载resnet50的权重
# --------------------------------#
model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


# -----------------------------------------------#
# 此处为定义3*3的卷积，即为指此次卷积的卷积核的大小为3*3
# -----------------------------------------------#
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


# -----------------------------------------------#
# 此处为定义1*1的卷积，即为指此次卷积的卷积核的大小为1*1
# -----------------------------------------------#
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# ----------------------------------#
# 此为resnet50中标准残差结构的定义
# conv3x3以及conv1x1均在该结构中被定义
# ----------------------------------#
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        # --------------------------------------------#
        # 当不指定正则化操作时将会默认进行二维的数据归一化操作
        # --------------------------------------------#
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # ---------------------------------------------------#
        # 根据input的planes确定width,width的值为
        # 卷积输出通道以及BatchNorm2d的数值
        # 因为在接下来resnet结构构建的过程中给到的planes的数值不相同
        # ---------------------------------------------------#
        width = int(planes * (base_width / 64.)) * groups
        # -----------------------------------------------#
        # 当步长的值不为1时,self.conv2 and self.downsample
        # 的作用均为对输入进行下采样操作
        # 下面为定义了一系列操作,包括卷积，数据归一化以及relu等
        # -----------------------------------------------#
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    # --------------------------------------#
    # 定义resnet50中的标准残差结构的前向传播函数
    # --------------------------------------#
    def forward(self, x):
        identity = x
        # -------------------------------------------------------------------------#
        # conv1*1->bn1->relu 先进行一次1*1的卷积之后进行数据归一化操作最后过relu增加非线性因素
        # conv3*3->bn2->relu 先进行一次3*3的卷积之后进行数据归一化操作最后过relu增加非线性因素
        # conv1*1->bn3 先进行一次1*1的卷积之后进行数据归一化操作
        # -------------------------------------------------------------------------#
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # -----------------------------#
        # 若有下采样操作则进行一次下采样操作
        # -----------------------------#
        if self.downsample is not None:
            identity = self.downsample(identity)
        # ---------------------------------------------#
        # 首先是将两部分进行add操作,最后过relu来增加非线性因素
        # concat（堆叠）可以看作是通道数的增加
        # add（相加）可以看作是特征图相加，通道数不变
        # add可以看作特殊的concat,并且其计算量相对较小
        # ---------------------------------------------#
        out += identity
        out = self.relu(out)

        return out


# --------------------------------#
# 此为resnet50网络的定义
# input的大小为224*224
# 初始化函数中的block即为上面定义的
# 标准残差结构--Bottleneck
# --------------------------------#
class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, max_elem=32, d_model=64):

        super(ResNet, self).__init__()
        self.num_classes = max_elem * d_model
        self.max_elem = max_elem
        self.d_model = d_model
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        # ---------------------------------------------------------#
        # 使用膨胀率来替代stride,若replace_stride_with_dilation为none
        # 则这个列表中的三个值均为False
        # ---------------------------------------------------------#
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        # ----------------------------------------------#
        # 若replace_stride_with_dilation这个列表的长度不为3
        # 则会有ValueError
        # ----------------------------------------------#
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.block = block
        self.groups = groups
        self.base_width = width_per_group
        # -----------------------------------#
        # conv1*1->bn1->relu
        # 224,224,3 -> 112,112,64
        # -----------------------------------#
        self.conv1 = nn.Conv2d(4, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # ------------------------------------#
        # 最大池化只会改变特征图像的高度以及
        # 宽度,其通道数并不会发生改变
        # 112,112,64 -> 56,56,64
        # ------------------------------------#
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 56,56,64   -> 56,56,256
        self.layer1 = self._make_layer(block, 64, layers[0])

        # 56,56,256  -> 28,28,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])

        # 28,28,512  -> 14,14,1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])

        # 14,14,1024 -> 7,7,2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        # --------------------------------------------#
        # 自适应的二维平均池化操作,特征图像的高和宽的值均变为1
        # 并且特征图像的通道数将不会发生改变
        # 7,7,2048 -> 1,1,2048
        # --------------------------------------------#
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # ----------------------------------------#
        # 将目前的特征通道数变成所要求的特征通道数（1000）
        # 2048 -> num_classes
        # ----------------------------------------#
        self.fc = nn.Linear(512 * block.expansion, self.num_classes)

        # -------------------------------#
        # 部分权重的初始化操作
        # -------------------------------#
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # -------------------------------#
        # 部分权重的初始化操作
        # -------------------------------#
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    # --------------------------------------#
    # _make_layer这个函数的定义其可以在类的
    # 初始化函数中被调用
    # block即为上面定义的标准残差结构--Bottleneck
    # --------------------------------------#
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        # -----------------------------------#
        # 在函数的定义中dilate的值为False
        # 所以说下面的语句将直接跳过
        # -----------------------------------#
        if dilate:
            self.dilation *= stride
            stride = 1
        # -----------------------------------------------------------#
        # 如果stride！=1或者self.inplanes != planes * block.expansion
        # 则downsample将有一次1*1的conv以及一次BatchNorm2d
        # -----------------------------------------------------------#
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        # -----------------------------------------------#
        # 首先定义一个layers,其为一个列表
        # 卷积块的定义,每一个卷积块可以理解为一个Bottleneck的使用
        # -----------------------------------------------#
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # identity_block
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    # ------------------------------#
    # resnet50的前向传播函数
    # ------------------------------#
    def forward(self, x):
        b_s = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # --------------------------------------#
        # 按照x的第1个维度拼接（按照列来拼接，横向拼接）
        # 拼接之后,张量的shape为(batch_size,2048)
        # --------------------------------------#
        x = torch.flatten(x, 1)
        # --------------------------------------#
        # 过全连接层来调整特征通道数
        # (batch_size,2048)->(batch_size,1000)
        # --------------------------------------#
        # print(x.shape)
        x = self.fc(x)

        x = x.view(b_s, self.max_elem, self.d_model)
        return x


# F = torch.randn(1, 4, 350, 240)
# print("As begin,shape:", format(F.shape))
# resnet = ResNet(Bottleneck, [3, 4, 6, 3])
# F = resnet(F)
# print(F.shape)