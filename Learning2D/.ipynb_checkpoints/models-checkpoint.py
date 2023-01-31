import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
import timm

class TimmModel(nn.Module):
    def __init__(self, backbone, bs, in_chans,  pretrained=False):
        super(TimmModel, self).__init__()
        self.in_chans = in_chans
        self.encoder = timm.create_model(
            backbone,
            in_chans=in_chans,
            num_classes=1,
            features_only=False,
            drop_rate=0.,
            drop_path_rate=0.,
            pretrained=pretrained
        )

        
        hdim = self.encoder.conv_head.out_channels
        self.encoder.classifier = nn.Identity()

        self.lstm = nn.LSTM(hdim, 256, num_layers=2, dropout=0., bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, x):  # (bs, nslice, ch, sz, sz)
        bs = x.shape[0]
        x = x.view(bs * 98, self.in_chans, 100, 100)
        feat = self.encoder(x)
        feat = feat.view(bs, 98, -1)
        feat, _ = self.lstm(feat)
        feat = feat.contiguous().view(bs * 98, -1)
        feat = self.head(feat)
        #feat = feat.view(bs, 98).contiguous()

        return feat