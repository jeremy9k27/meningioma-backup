# %%
import torch.nn as nn

class BottleneckResid3d(nn.Module):
    def __init__(self, input_channels, num_filters, kernel_size, dropout_rate, activation):
        super(BottleneckResid3d, self).__init__()
        # shortcut for residual learning
        self.shortcut = nn.Conv3d(in_channels=input_channels, out_channels=num_filters, kernel_size=1) if input_channels != num_filters else nn.Identity()
        # first 1x1x1 conv for bottleneck block using 1/4 filters
        self.bn1 = nn.BatchNorm3d(input_channels)
        self.act1 = nn.LeakyReLU()
        self.conv1 = nn.Conv3d(in_channels=input_channels, out_channels=int(round(num_filters / 4)), kernel_size=1)
        # 3x3x3 conv for bottleneck block with bn and activation using 1/4 filters
        self.bn2 = nn.BatchNorm3d(int(round(num_filters / 4)))
        self.act2 = nn.LeakyReLU()
        self.conv2 = nn.Conv3d(in_channels=int(round(num_filters / 4)), out_channels=int(round(num_filters / 4)), kernel_size=kernel_size, padding=kernel_size//2)
        # second 1x1x1 conv with full filters (no strides)
        self.bn3 = nn.BatchNorm3d(int(round(num_filters / 4)))
        self.act3 = nn.LeakyReLU()
        self.conv3 = nn.Conv3d(in_channels=int(round(num_filters / 4)), out_channels=num_filters, kernel_size=1)
        # optional dropout
        self.dropout = nn.Dropout3d(p=dropout_rate) if dropout_rate else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.bn1(x)
        out = self.act1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv3(out)
        out = self.dropout(out)
        out += residual
        return out

class CalabreseModel(nn.Module):
    def __init__(self, original_shape=96, layer_layout=[1, 1, 2, 2], input_channels=11, base_filters=32, dropout_rate=0.4, kernel_size=3, activation="leaky_relu", dense_features=32, output_features=1, final_layer="sigmoid"):
        super(CalabreseModel, self).__init__()

        # we use nn.ModuleList to build our model dynamically
        self.layers = nn.ModuleList()
        in_chan = input_channels
        num_filters = base_filters
        # loop to create all the bottleneck residual 3d blocks, with 3d max pooling at end of each level
        for i, level in enumerate(layer_layout, 1):
            for block in range(level):
                self.layers.append(BottleneckResid3d(input_channels=in_chan, num_filters=num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate, activation=activation))
            self.layers.append(nn.MaxPool3d(kernel_size=2))
            in_chan = num_filters
            num_filters = int(num_filters*2)
        # flatten and have final dense layer
        self.layers.append(nn.Flatten())
        # Calculate the number of input features for the linear layer
        self.layers.append(nn.Linear(int(num_filters*(original_shape/(2**len(layer_layout)))**3), dense_features))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Linear(dense_features, output_features))
        self.layers.append(nn.Sigmoid())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
m = CalabreseModel()
# %%
