import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelBlock(nn.Module):
    def __init__(self, block):
        super(ModelBlock, self).__init__()
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class RecursiveNN_Linear(nn.Module):
    def __init__(
        self, in_features, N1, N2, out_features, leaky_relu=False, dropout=0.5
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.activation = F.leaky_relu if leaky_relu else F.relu
        self.activation_kwargs = {"negative_slope": 0.4} if leaky_relu else {}

        self.linear1a = nn.Linear(in_features=in_features, out_features=N1, bias=True)
        self.linear2a = nn.Linear(in_features=N1, out_features=N2, bias=True)

        self.linear3a = nn.Linear(in_features=N2, out_features=out_features, bias=True)

        self.linear1b = nn.Linear(
            in_features=10 + out_features, out_features=N1, bias=True
        )

        self.linear2b = nn.Linear(in_features=N1, out_features=N2, bias=True)

        self.linear3b = nn.Linear(in_features=N2, out_features=1, bias=True)

    def forward(self, laser_inputs, other_features):

        out = self.linear1a(laser_inputs)
        out = self.activation(out, **self.activation_kwargs)
        out = self.dropout(out)

        out = self.linear2a(out)
        out = self.activation(out, **self.activation_kwargs)
        out = self.dropout(out)

        out = self.linear3a(out)
        out = self.activation(out, **self.activation_kwargs)

        out = torch.cat((out, other_features), dim=1)

        out = self.linear1b(out)
        out = self.activation(out, **self.activation_kwargs)
        out = self.dropout(out)

        out = self.linear2b(out)
        out = self.activation(out, **self.activation_kwargs)
        out = self.dropout(out)

        out = self.linear3b(out)

        return out.view(-1)


class RecursiveNN(nn.Module):
    def __init__(self, ModelBlock, conv_diction, ffnn_diction, BASELINE_dim=10):
        super(RecursiveNN, self).__init__()

        self.BASELINE_dim = BASELINE_dim
        self.Hin = [1024]
        self.Hout = []
        self.Win = [1]

        # Convolution Variables
        self.conv_dict = conv_diction

        self.Kszes = conv_diction["Ksze"]
        self.InChannels = conv_diction["InChannels"]
        self.OutChannels = conv_diction["OutChannels"]
        self.Strides = conv_diction["Stride"]
        self.Paddings = conv_diction["Padding"]

        self.PoolingDim = self.conv_dict.pop("MaxPoolDim")
        self.PoolingBool = self.conv_dict.pop("MaxPoolBool")

        # FFNN Variables
        self.ffnn_dict = ffnn_diction

        self.hidden_laser = ffnn_diction["laser_hidden_layers"]
        self.hidden_mixture = ffnn_diction["mixture_hidden_layers"]

        # Convolution of LASER embeddings
        conv_seq = self.make_conv_layer(ModelBlock)

        # FFNN
        # Joint features vector from LASER embeddings and BASELINE features
        ffnn_seq = self.make_ffnn_layer(ModelBlock)

        # Define the model
        self.conv_seq = conv_seq
        self.ffnn_seq = ffnn_seq

        # PRINT TO VISUALISE DURING INITIALISATION
        # print(self.conv_seq)
        # print()
        # print(self.ffnn_seq)

    def make_conv_layer(self, ModelBlock):
        layers = []

        # Create a fully convolutional layer
        for idx in range(len(self.Strides)):

            self.Hout.append(
                int(
                    (self.Hin[idx] - self.Kszes[idx] + 2 * self.Paddings[idx])
                    / (self.Strides[idx])
                    + 1
                )
            )
            if idx is not len(self.Strides):
                self.Hin.append(int(self.Hout[idx]))

            layer_subset = [self.conv_dict[feat][idx] for feat in self.conv_dict.keys()]
            block = [
                nn.Conv1d(*layer_subset),
                nn.BatchNorm1d(self.OutChannels[idx]),
                nn.LeakyReLU(inplace=True, negative_slope=0.2),
            ]
            module_block = ModelBlock(block)
            layers.append(module_block)

        if self.PoolingBool:
            layers.append(nn.MaxPool1d(self.PoolingDim, self.PoolingDim))
            self.Hout.append(self.Hout[-1] / self.PoolingDim)

        nfc = int(self.Hout[-1]) * int(self.OutChannels[-1])
        self.hidden_laser.insert(0, nfc)

        layers.append(View((-1, nfc)))

        # Now make a FFNN from convolutional layer output into latent space size
        for idx in range(len(self.hidden_laser) - 1):
            block = [
                nn.Linear(
                    self.hidden_laser[idx], self.hidden_laser[idx + 1], bias=True
                ),
                nn.LeakyReLU(inplace=True, negative_slope=0.6),
                nn.Dropout(p=0.4),
            ]
            module_block = ModelBlock(block)
            layers.append(module_block)
        # layers.append(ModelBlock([nn.Tanh()]))
        return nn.Sequential(*layers)

    def make_ffnn_layer(self, ModelBlock):
        layers = []

        # Add the mixture FFNN combining baseline with convolution from LASER
        for idx in range(len(self.hidden_mixture) - 1):

            if idx == 0:
                block = [
                    nn.Linear(
                        self.hidden_laser[-1] + self.BASELINE_dim,
                        self.hidden_mixture[idx],
                        bias=True,
                    ),
                    nn.LeakyReLU(inplace=True, negative_slope=0.8),
                    nn.Dropout(p=0.4),
                ]
            else:
                block = [
                    nn.Linear(
                        self.hidden_mixture[idx - 1],
                        self.hidden_mixture[idx],
                        bias=True,
                    ),
                    nn.LeakyReLU(inplace=True, negative_slope=0.8),
                    nn.Dropout(p=0.2),
                ]

            module_block = ModelBlock(block)
            layers.append(module_block)

        block = [
            nn.Linear(self.hidden_mixture[idx], self.hidden_mixture[idx + 1], bias=True)
        ]
        module_block = ModelBlock(block)
        layers.append(module_block)

        return nn.Sequential(*layers)

    def forward(self, laser_inputs, baseline_features):

        out = self.conv_seq(laser_inputs)
        out = torch.cat((out, baseline_features), dim=1)
        out = self.ffnn_seq(out)

        # out = self.ffnn_seq(baseline_features)
        return out.view(-1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
