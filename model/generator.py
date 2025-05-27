import torch
import torch.nn as nn
from .blocks import ConditionalBatchNorm2d, ResBlock
from config import image_size, num_classes

class Generator(nn.Module):
    def __init__(self, z_dim, num_classes, img_channels=3):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.init_size = image_size // 4
        self.l1 = nn.Sequential(nn.Linear(z_dim + num_classes, 128 * self.init_size ** 2))

        self.cbns = nn.ModuleList([
            ConditionalBatchNorm2d(128, num_classes),
            ConditionalBatchNorm2d(128, num_classes),
            ConditionalBatchNorm2d(64, num_classes)
        ])

        self.conv_blocks = nn.ModuleList([
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.Conv2d(64, img_channels, 3, padding=1)
        ])

        self.res1 = ResBlock(128)
        self.res2 = ResBlock(64)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, z, labels):
        gen_input = torch.cat((z, self.label_emb(labels)), dim=1)
        out = self.l1(gen_input)
        out = out.view(z.size(0), 128, self.init_size, self.init_size)

        out = self.cbns[0](out, labels)
        out = self.upsample(out)
        out = nn.ReLU(inplace=True)(self.conv_blocks[0](out))

        out = self.cbns[1](out, labels)
        out = self.upsample(out)
        out = nn.ReLU(inplace=True)(self.conv_blocks[1](out))

        out = self.cbns[2](out, labels)
        out = self.res2(out)
        out = torch.tanh(self.conv_blocks[2](out))
        return out
