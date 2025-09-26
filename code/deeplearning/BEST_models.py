import torch.nn as nn
import torch.nn.functional as F
import torch

class BottleneckResid3d(nn.Module):
    def __init__(self, input_channels, num_filters, kernel_size, dropout_rate, activation, use_batch = True):
        super(BottleneckResid3d, self).__init__()
        # shortcut for residual learning
        self.shortcut = nn.Conv3d(in_channels=input_channels, out_channels=num_filters, kernel_size=1) if input_channels != num_filters else nn.Identity()
        # first 1x1x1 conv for bottleneck block using 1/4 filters
        self.use_batch = use_batch
        if use_batch:
            self.norm1 = nn.BatchNorm3d(input_channels)
        else:
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=input_channels)
        self.act1 = nn.LeakyReLU()
        self.conv1 = nn.Conv3d(in_channels=input_channels, out_channels=int(round(num_filters / 4)), kernel_size=1)
        # 3x3x3 conv for bottleneck block with bn and activation using 1/4 filters
        if use_batch: 
            self.norm2 = nn.BatchNorm3d(int(round(num_filters / 4)))
        else:
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=int(round(num_filters / 4)))
        self.act2 = nn.LeakyReLU()
        self.conv2 = nn.Conv3d(in_channels=int(round(num_filters / 4)), out_channels=int(round(num_filters / 4)), kernel_size=kernel_size, padding=kernel_size//2)
        # second 1x1x1 conv with full filters (no strides)
        if use_batch:
            self.norm3 = nn.BatchNorm3d(int(round(num_filters / 4)))
        else:
            self.norm3 = nn.GroupNorm(num_groups=8, num_channels=int(round(num_filters / 4)))
        self.act3 = nn.LeakyReLU()
        self.conv3 = nn.Conv3d(in_channels=int(round(num_filters / 4)), out_channels=num_filters, kernel_size=1)
        # optional dropout
        self.dropout = nn.Dropout3d(p=dropout_rate) if dropout_rate else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.norm1(x)
        out = self.act1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.conv2(out)
        out = self.norm3(out)
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
                in_chan = num_filters
            self.layers.append(nn.MaxPool3d(kernel_size=2))
            num_filters = int(num_filters*2)
        # flatten and have final dense layer
        self.layers.append(nn.Flatten())
        # Calculate the number of input features for the linear layer

        self.layers.append(nn.LazyLinear(dense_features))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Linear(dense_features, output_features))
        self.layers.append(nn.Sigmoid())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_num_trainable_params(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable_params

class CalabreseModelEncoder(nn.Module):
    def __init__(self, original_shape=96, layer_layout=[1, 1, 2, 2], input_channels=11, base_filters=32, dropout_rate=0.4, kernel_size=3, activation="leaky_relu", dense_features=32, output_features=1, final_layer="sigmoid", pyrad_targets = 1):
        super(CalabreseModelEncoder, self).__init__()

        # we use nn.ModuleList to build our model dynamically
        self.encoder = nn.ModuleList()
        in_chan = input_channels
        num_filters = base_filters
        # loop to create all the bottleneck residual 3d blocks, with 3d max pooling at end of each level
        for i, level in enumerate(layer_layout, 1):
            for block in range(level):
                self.encoder.append(BottleneckResid3d(input_channels=in_chan, num_filters=num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate, activation=activation))
                in_chan = num_filters
            self.encoder.append(nn.MaxPool3d(kernel_size=2))
            num_filters = int(num_filters*2)

        self.classifier = nn.ModuleList()
        # flatten and have final dense layer
        self.classifier.append(nn.Flatten())
        # Calculate the number of input features for the linear layer
        size = int(0.5*num_filters*(original_shape/(2**len(layer_layout)))**3)
        
        self.classifier.append(nn.Linear(int(0.5*num_filters*(original_shape/(2**len(layer_layout)))**3), dense_features))
        self.classifier.append(nn.LeakyReLU())
        self.classifier.append(nn.Linear(dense_features, output_features))
        self.classifier.append(nn.Sigmoid())


        self.decoder = nn.ModuleList()
        for i in reversed(range(len(layer_layout))):
            # Mirror of encoder layers; adjust filters accordingly
            num_filters = int(base_filters * (2 ** i))
            self.decoder.append(nn.ConvTranspose3d(in_chan, num_filters, kernel_size=2, stride=2))
            self.decoder.append(BottleneckResid3d(input_channels=num_filters, num_filters=num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate, activation=activation))
            in_chan = num_filters

        # Final layer to map back to input_channels
        self.decoder.append(nn.Conv3d(in_chan, input_channels, kernel_size=1))

        # 1 layer pyrad mlp
        self.mlp_1_layer = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(pyrad_targets)
        )
    
        # 2 layer pyrad mlp
        self.mlp_2_layer = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Linear(512, pyrad_targets)
        )
    
    def forward_encoder(self, x):

        for layer in self.encoder:
            x = layer(x)
         
        return x

    def forward_autoencoder(self, x):
        z = self.forward_encoder(x)
        for layer in self.decoder:
            z = layer(z)
        return z

    def forward_pyrad_1layer(self, x):
        z = self.forward_encoder(x)
        for layer in self.mlp_1_layer:
            z = layer(z)
        return z

    def forward_pyrad_2layer(self, x):
        z = self.forward_encoder(x)
        for layer in self.mlp_2_layer:
            z = layer(z)
        return z

    def forward(self, x):
        z = self.forward_encoder(x)
        for layer in self.classifier:
            z = layer(z)
        return z
    
    def get_num_trainable_params(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable_params


class Vanilla3DAutoencoder(nn.Module):
    def __init__(self, in_channels=2, latent_dim=64):
        super(Vanilla3DAutoencoder, self).__init__()

        self.first = True
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 4, kernel_size=3, stride=2, padding=1),  # D/2
            nn.ReLU(),

            nn.Conv3d(4, 8, kernel_size=3, stride=2, padding=1),  # D/4
            nn.ReLU(),

            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),  # D/8
            nn.ReLU()
        )

        # Decoder (D/8 → D)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # D/4
            nn.ReLU(),

            nn.ConvTranspose3d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1),  # D/2
            nn.ReLU(),

            nn.ConvTranspose3d(4, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # D
            
            nn.Sigmoid()
        )

    def forward(self, x):
        #print("Input shape:", x.shape)
        z = self.encoder(x)
        if self.first: print("Latent shape:", z.shape)
        x_hat = self.decoder(z)
        #print("Output shape:", x_hat.shape)
        self.first = False
        
        return x_hat


class MLPClassifier(nn.Module):
    def __init__(self, encoder_output_shape=(32, 12, 12, 12), dense_features=32, output_features=1):
        super().__init__()
        num_flattened = 1
        for dim in encoder_output_shape:
            num_flattened *= dim
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_flattened, dense_features),
            nn.LeakyReLU(),
            nn.Linear(dense_features, output_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)