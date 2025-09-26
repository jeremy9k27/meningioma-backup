for i, level in enumerate(layer_layout, 1):             
    for block in range(level):                 
        self.encoder.append(BottleneckResid3d(input_channels=in_chan, num_filters=num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate, activation=activation, use_batch=use_batch))                 
        in_chan = num_filters             
        self.encoder.append(nn.MaxPool3d(kernel_size=2))             
        num_filters = int(num_filters*2) 
        
        self.decoder = nn.ModuleList()
        for i in reversed(range(len(layer_layout))):
            # Mirror of encoder layers; adjust filters accordingly
            num_filters = int(base_filters * (2 ** i))
            self.decoder.append(nn.ConvTranspose3d(in_chan, num_filters, kernel_size=2, stride=2))
            self.decoder.append(BottleneckResid3d(input_channels=num_filters, num_filters=num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate, activation=activation))
            in_chan = num_filters