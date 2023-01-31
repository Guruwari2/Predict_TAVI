import monai


model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, n_classes=2, inplanes = (64,128,256,512)).to('cuda')
