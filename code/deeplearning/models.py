import torch.nn as nn
import torch.nn.functional as F
import torch

class BottleneckResid3dReverse(nn.Module):
    def __init__(self, input_channels, num_filters, kernel_size, dropout_rate, activation, use_batch = True):
        super(BottleneckResid3dReverse, self).__init__()
        # shortcut for residual learning
        self.shortcut = nn.Conv3d(in_channels=input_channels, out_channels=num_filters, kernel_size=1) if input_channels != num_filters else nn.Identity()
        # first 1x1x1 conv for bottleneck block using 1/4 filters
        self.use_batch = use_batch
        if use_batch:
            self.norm1 = nn.BatchNorm3d(input_channels)
        else:
            self.norm1 = nn.GroupNorm(num_groups=2, num_channels=input_channels)
        self.act1 = nn.LeakyReLU()
        self.conv1 = nn.Conv3d(in_channels=input_channels, out_channels=num_filters*4, kernel_size=1)
        # 3x3x3 conv for bottleneck block with bn and activation using 1/4 filters
        if use_batch: 
            self.norm2 = nn.BatchNorm3d(num_filters*4)
        else:
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=num_filters*4)
        self.act2 = nn.LeakyReLU()
        self.conv2 = nn.Conv3d(in_channels=num_filters*4, out_channels=num_filters*4, kernel_size=kernel_size, padding=kernel_size//2)
        # second 1x1x1 conv with full filters (no strides)
        if use_batch:
            self.norm3 = nn.BatchNorm3d(num_filters*4)
        else:
            self.norm3 = nn.GroupNorm(num_groups=8, num_channels=num_filters*4)
        self.act3 = nn.LeakyReLU()
        self.conv3 = nn.Conv3d(in_channels=num_filters*4, out_channels=num_filters, kernel_size=1)
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
            self.norm1 = nn.GroupNorm(num_groups=input_channels, num_channels=input_channels)
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

class DenseBN(nn.Module):
    def __init__(self, in_features, out_features, use_batch, dropout=0.0):
        super(DenseBN, self).__init__()
        self.fc = nn.Linear(in_features, out_features)  
        if use_batch:
            self.bn = nn.BatchNorm1d(out_features)
        else:
            self.bn = nn.GroupNorm(num_groups=1, num_channels=out_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.act = nn.LeakyReLU()
            

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

class CalabreseModelUNetSkip(nn.Module):
    def __init__(self, use_batch, original_shape=96, layer_layout=[1, 1, 2, 2], input_channels=11, base_filters=32, dropout_rate=0.4, kernel_size=3, activation="leaky_relu",dense_features=32, output_features=1, final_layer="sigmoid", pyrad_targets = 1):
        super(CalabreseModelUNetSkip, self).__init__()

        # we use nn.ModuleList to build our model dynamically
        self.encoder = nn.ModuleList()
        in_chan = input_channels
        num_filters = base_filters
        # loop to create all the bottleneck residual 3d blocks, with 3d max pooling at end of each level
        for i, level in enumerate(layer_layout, 1):    
            for block in range(level):
                #print("encoder", i, in_chan, num_filters)
                self.encoder.append(BottleneckResid3d(input_channels=in_chan, num_filters=num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate, activation=activation, use_batch=use_batch))
                in_chan = num_filters
            
            #print("max pool")    
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
        decoder_in_chan = in_chan
        decoder_num_filters = num_filters // 2 
        

        for i, level in enumerate(reversed(layer_layout), 1):
            #print("inv conv:",  decoder_in_chan)
            self.decoder.append(nn.ConvTranspose3d(decoder_in_chan, decoder_in_chan, kernel_size=2, stride=2))
            
            for block in range(level):
                

                
                if level==1: #one block in the level, filters need to be halved, has to accept concatted input

                    decoder_num_filters = decoder_num_filters // 2
                    if i == len(layer_layout): # if last last layer
                        decoder_num_filters = input_channels
                    self.decoder.append(BottleneckResid3dReverse(input_channels=decoder_in_chan*2, num_filters=decoder_num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate, activation=activation, use_batch=use_batch))
                
                elif level > 1 and block == level-1: # if last block in the level, the filters need to be halved, will not recieve concatted
                    
                    if i == len(layer_layout):
                        decoder_num_filters = input_channels
                    else:
                        decoder_num_filters = decoder_num_filters // 2
                    #print("decoder", i, decoder_in_chan, decoder_num_filters)
                    self.decoder.append(BottleneckResid3dReverse(input_channels=decoder_in_chan, num_filters=decoder_num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate, activation=activation, use_batch=use_batch))

                elif level > 1 and block < level-1: # accept concatted
                    self.decoder.append(BottleneckResid3dReverse(input_channels=decoder_in_chan*2, num_filters=decoder_num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate, activation=activation, use_batch=use_batch))
                
                decoder_in_chan = decoder_num_filters

        self.seg = nn.ModuleList()
        self.seg.append(nn.Conv3d(in_channels=2, out_channels=4, kernel_size=1))
            

        '''
        # 1 layer pyrad mlp
        self.mlp_1_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(55296, pyrad_targets)
        )
    
        
        # 2 layer pyrad mlp
        self.mlp_2_layer = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # Reduces to (batch, 256, 1, 1, 1)
            nn.Flatten(),             # Now only 256 features
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(128, pyrad_targets)
        )
        '''

    def forward_encoder(self, x):
        skips = []
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.MaxPool3d):
                skips.append(x)  
                x = layer(x)    
            else:
                x = layer(x)    

        return (x, skips)

    def forward_seg(self, x):

        x, skips = self.forward_encoder(x)
        skip_idx = 0
        skips.reverse()
        
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.ConvTranspose3d):
                x = layer(x)
                #print(x.shape, skips[skip_idx].shape)
                x = torch.cat([x, skips[skip_idx]], dim=1)
                skip_idx += 1

            else:
                x = layer(x)
        
        for layer in self.seg:
            x = layer(x)
            
        return x        
        
    def forward_autoencoder(self, x):
        x, skips = self.forward_encoder(x)
        skip_idx = 0
        skips.reverse()
        
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.ConvTranspose3d):
                x = layer(x)
                #print(x.shape, skips[skip_idx].shape)
                x = torch.cat([x, skips[skip_idx]], dim=1)
                skip_idx += 1

            else:
                x = layer(x)
            
        return x
    
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

class CalabreseModelEncoder(nn.Module):
    def __init__(self, use_batch, original_shape=96, layer_layout=[1, 1, 2, 2], input_channels=11, base_filters=32, dropout_rate=0.4, kernel_size=3, activation="leaky_relu",dense_features=32, output_features=1, final_layer="sigmoid", pyrad_targets = 1):
        super(CalabreseModelEncoder, self).__init__()

        # we use nn.ModuleList to build our model dynamically
        self.encoder = nn.ModuleList()
        in_chan = input_channels
        num_filters = base_filters
        # loop to create all the bottleneck residual 3d blocks, with 3d max pooling at end of each level
        for i, level in enumerate(layer_layout, 1):    
            for block in range(level):
                #print("encoder", i, in_chan, num_filters)
                self.encoder.append(BottleneckResid3d(input_channels=in_chan, num_filters=num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate, activation=activation, use_batch=use_batch))
                in_chan = num_filters
            
            #print("max pool")    
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

        '''
        self.decoder = nn.ModuleList()
        decoder_in_chan = in_chan
        decoder_num_filters = num_filters // 2 
        print(decoder_num_filters)

        for i, level in enumerate(reversed(layer_layout), 1):
            #print("inv conv:",  decoder_in_chan)
            self.decoder.append(nn.ConvTranspose3d(decoder_in_chan, decoder_in_chan, kernel_size=2, stride=2))
            
            for block in range(level):
                
                if block == level-1: # if last block in the level, the filters need to be halved
                    if i == len(layer_layout):
                        decoder_num_filters = input_channels
                    else:
                        decoder_num_filters = decoder_num_filters // 2
                    #print("decoder", i, decoder_in_chan, decoder_num_filters)
                    self.decoder.append(BottleneckResid3dReverse(input_channels=decoder_in_chan, num_filters=decoder_num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate, activation=activation, use_batch=use_batch))
                
                else:
                    #print("decoder", i, decoder_in_chan, decoder_num_filters)
                    self.decoder.append(BottleneckResid3dReverse(input_channels=decoder_in_chan, num_filters=decoder_num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate, activation=activation, use_batch=use_batch))
                
                decoder_in_chan = decoder_num_filters
        '''

        '''
        # 1 layer pyrad mlp
        self.mlp_1_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(55296, pyrad_targets)
        )
        
        
        # 2 layer pyrad mlp
        self.mlp_2_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(55296, 128),
            nn.LeakyReLU(),
            nn.Linear(128, pyrad_targets)
        )
        '''
        

    def forward_encoder(self, x):

        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.MaxPool3d):
                x = layer(x)    
            else:
                x = layer(x)    

        return x

    def forward_autoencoder(self, x):
        x= self.forward_encoder(x)

        
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.ConvTranspose3d):
                x = layer(x)

            else:
                x = layer(x)
            
        return x
    
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

class CalabreseModelExact(nn.Module):
    def __init__(self, use_batch, original_shape=96, layer_layout=[1, 1, 2, 2], input_channels=11, base_filters=32, dropout_rate=0.4, kernel_size=3, activation="leaky_relu",dense_features=32, output_features=1, final_layer="sigmoid", pyrad_targets = 1):
        super(CalabreseModelExact, self).__init__()

        # we use nn.ModuleList to build our model dynamically
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Conv3d(in_channels=input_channels, out_channels=base_filters, kernel_size=kernel_size, padding=kernel_size//2))
        in_chan = base_filters
        num_filters = base_filters*2
        # loop to create all the bottleneck residual 3d blocks, with 3d max pooling at end of each level
        for i, level in enumerate(layer_layout, 1):    
            for block in range(level):
                #print("encoder", i, in_chan, num_filters)
                self.encoder.append(BottleneckResid3d(input_channels=in_chan, num_filters=num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate, activation=activation, use_batch=use_batch))
                in_chan = num_filters
            
            #print("max pool")    
            self.encoder.append(nn.MaxPool3d(kernel_size=2))
            num_filters = int(num_filters*2)
        
        #self.encoder = nn.ModuleList(list(self.encoder)[:-1])
        #self.encoder.append(nn.MaxPool3d(kernel_size=3))
        
        self.classifier = nn.ModuleList()
        self.classifier.append(nn.Flatten())  
        size = int(0.5*num_filters*(original_shape/(2**len(layer_layout)))**3)
        print("size", size)
        self.classifier.append(nn.Linear(size, dense_features))
        self.classifier.append(nn.LeakyReLU())  
        self.classifier.append(nn.Linear(dense_features, output_features))
        self.classifier.append(nn.Sigmoid())


        

    def forward_encoder(self, x):

        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.MaxPool3d):
                x = layer(x) 
                #print(x.shape) 
            else:
                x = layer(x)    

        return x
    

    def forward(self, x):
        z = self.forward_encoder(x)
        for layer in self.classifier:
            z = layer(z)
        return z
    
    def get_num_trainable_params(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable_params


class CalabreseModelPool(nn.Module):
    def __init__(self, use_batch, original_shape=96, layer_layout=[1, 1, 2, 2], input_channels=11, base_filters=32, dropout_rate=0.4, kernel_size=3, activation="leaky_relu",dense_features=32, output_features=1, final_layer="sigmoid", pyrad_targets = 1):
        super(CalabreseModelPool, self).__init__()
        
        # we use nn.ModuleList to build our model dynamically
        self.encoder = nn.ModuleList()
        in_chan = input_channels
        num_filters = base_filters
        # loop to create all the bottleneck residual 3d blocks, with 3d max pooling at end of each level
        for i, level in enumerate(layer_layout, 1):    
            for block in range(level):
                #print("encoder", i, in_chan, num_filters)
                self.encoder.append(BottleneckResid3d(input_channels=in_chan, num_filters=num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate, activation=activation, use_batch=use_batch))
                in_chan = num_filters
            
            #print("max pool")    
            self.encoder.append(nn.MaxPool3d(kernel_size=2))
            num_filters = int(num_filters*2)

        self.encoder = nn.ModuleList(list(self.encoder)[:-1])
        self.encoder.append(nn.MaxPool3d(kernel_size=3))

        self.classifier = nn.ModuleList()
        # flatten and have final dense layer
        self.classifier.append(nn.Flatten())
        # Calculate the number of input features for the linear layer
        size = int(0.5*num_filters*(original_shape/(2**len(layer_layout)))**3)
        
        self.classifier.append(nn.Linear(16384, dense_features))
        self.classifier.append(nn.LeakyReLU())
        self.classifier.append(nn.Linear(dense_features, output_features))
        self.classifier.append(nn.Sigmoid())



        '''
        self.decoder = nn.ModuleList()
        decoder_in_chan = in_chan
        decoder_num_filters = num_filters // 2 
        print(decoder_num_filters)

        for i, level in enumerate(reversed(layer_layout), 1):
            #print("inv conv:",  decoder_in_chan)
            self.decoder.append(nn.ConvTranspose3d(decoder_in_chan, decoder_in_chan, kernel_size=2, stride=2))
            
            for block in range(level):
                
                if block == level-1: # if last block in the level, the filters need to be halved
                    if i == len(layer_layout):
                        decoder_num_filters = input_channels
                    else:
                        decoder_num_filters = decoder_num_filters // 2
                    #print("decoder", i, decoder_in_chan, decoder_num_filters)
                    self.decoder.append(BottleneckResid3dReverse(input_channels=decoder_in_chan, num_filters=decoder_num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate, activation=activation, use_batch=use_batch))
                
                else:
                    #print("decoder", i, decoder_in_chan, decoder_num_filters)
                    self.decoder.append(BottleneckResid3dReverse(input_channels=decoder_in_chan, num_filters=decoder_num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate, activation=activation, use_batch=use_batch))
                
                decoder_in_chan = decoder_num_filters
        '''

        
        

    def forward_encoder(self, x):

        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.MaxPool3d):
                x = layer(x)
                #print(x.shape)    
            else:
                x = layer(x)    

        return x

    def forward_autoencoder(self, x):
        x= self.forward_encoder(x)

        
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.ConvTranspose3d):
                x = layer(x)

            else:
                x = layer(x)
            
        return x
    

    def forward(self, x):
        z = self.forward_encoder(x)
        for layer in self.classifier:
            z = layer(z)
        return z
    
    def get_num_trainable_params(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable_params


class CalabreseModelFuse(nn.Module):
    def __init__(self, use_batch, original_shape=96, layer_layout=[1, 1, 2, 2], input_channels=11, base_filters=32, dropout_rate=0.4, kernel_size=3, activation="leaky_relu",dense_features=32, output_features=1, final_layer="sigmoid", n_pyrad_feats = 9):
        super(CalabreseModelFuse, self).__init__()

        # we use nn.ModuleList to build our model dynamically
        self.encoder = nn.ModuleList()
        in_chan = input_channels
        num_filters = base_filters
        # loop to create all the bottleneck residual 3d blocks, with 3d max pooling at end of each level
        for i, level in enumerate(layer_layout, 1):    
            for block in range(level):
                #print("encoder", i, in_chan, num_filters)
                self.encoder.append(BottleneckResid3d(input_channels=in_chan, num_filters=num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate, activation=activation, use_batch=use_batch))
                in_chan = num_filters
            
            #print("max pool")    
            self.encoder.append(nn.MaxPool3d(kernel_size=2))
            num_filters = int(num_filters*2)
        
        self.encoder.append(nn.Flatten())
        size = int(0.5*num_filters*(original_shape/(2**len(layer_layout)))**3)    
        self.encoder.append(nn.Linear(int(0.5*num_filters*(original_shape/(2**len(layer_layout)))**3), dense_features))
        self.encoder.append(nn.LeakyReLU())

        self.pyrad = nn.ModuleList()
        self.pyrad.append(DenseBN(n_pyrad_feats, n_pyrad_feats, dropout = dropout_rate, use_batch=False))
        self.pyrad.append(DenseBN(n_pyrad_feats, n_pyrad_feats*2, dropout = dropout_rate, use_batch=False))
        self.pyrad.append(DenseBN(n_pyrad_feats*2, n_pyrad_feats, dropout = dropout_rate, use_batch=False))


        self.classifier = nn.ModuleList()
        self.classifier.append(nn.Linear(dense_features+n_pyrad_feats, output_features))
        self.classifier.append(nn.Sigmoid())


    def forward(self, x, pyrad_feats):

        for layer in self.encoder:           
            x = layer(x)

        for layer in self.pyrad:
            pyrad_feats = layer(pyrad_feats)

        z = torch.cat([x, pyrad_feats], dim=1)

        for layer in self.classifier:
            z = layer(z)

        return z
    
    def get_num_trainable_params(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable_params
