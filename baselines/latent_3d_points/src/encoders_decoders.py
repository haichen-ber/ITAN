import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, k=3, n_filters=[64, 128, 128, 256, 128]):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.n_layer = len(n_filters)
        for i in range(self.n_layer):
            if i == 0:
                in_c = k
            else:
                in_c = n_filters[i-1]
            out_c = n_filters[i]
            conv = nn.Conv1d(in_c, out_c, 1)
            bn = nn.BatchNorm1d(out_c)
            self.conv_layers.append(nn.Sequential(conv, bn))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for i in range(self.n_layer):
            x = self.relu(self.conv_layers[i](x))
        x = torch.max(x, 2)[0]
        return x

class Decoder(nn.Module):
    def __init__(self, k=1024, layer_sizes=[256, 256, 1024*3]):
        super().__init__()
        self.fc_layers = nn.ModuleList()
        self.n_layer = len(layer_sizes)
        self.in_num = k
        for i in range(self.n_layer):
            if i == 0:
                in_c = k
            else:
                in_c = layer_sizes[i-1]
            out_c = layer_sizes[i]
            linear = nn.Linear(in_c, out_c)
            self.fc_layers.append(linear)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        bs = x.shape[0]
        for i in range(self.n_layer-1):
            x = self.relu(self.fc_layers[i](x))
        x = self.fc_layers[self.n_layer-1](x)
        x = x.view(bs, 3, -1)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.encoder = Encoder(k)
        self.decoder = Decoder(128)


    def forward(self, x):
        z = self.encoder(x)
        self.bottleneck_size = int(z.shape[1])
        x_reconstr = self.decoder(z)

        return x_reconstr



        

