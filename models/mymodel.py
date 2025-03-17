from .modelinstance.UNet import UNet


def mymodel(n_channels, n_classes):
    return UNet(n_channels, n_classes) 

    # return xxx