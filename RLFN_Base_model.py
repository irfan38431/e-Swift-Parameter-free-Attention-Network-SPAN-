# Import necessary libraries for model definition
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

# Utility function to ensure kernel size is a tuple of two integers
def _make_pair(value):
    """
    Helper function to convert a single integer to a tuple of two identical integers.
    This ensures that any kernel size or padding specified as a single integer 
    is applied equally across both dimensions (height and width).
    
    Parameters:
    - value: Either a single integer or a tuple of two integers.
    
    Returns:
    - A tuple of two identical integers.
    """
    if isinstance(value, int):
        value = (value,) * 2  # Converts single integer to a tuple (value, value)
    return value

# Function to create a convolutional layer with adaptive padding
def conv_layer(in_channels, out_channels, kernel_size, bias=True):
    """
    Creates a 2D convolutional layer with padding calculated based on the kernel size.
    This padding ensures that the input and output feature maps have the same spatial dimensions.

    Parameters:
    - in_channels: Number of channels in the input feature map.
    - out_channels: Number of filters in the convolution (output channels).
    - kernel_size: Size of the convolutional filter.
    - bias: Boolean flag indicating if a bias term should be added to the output.

    Returns:
    - A convolutional layer with the specified parameters.
    """
    kernel_size = _make_pair(kernel_size)  # Ensure kernel size is a tuple
    padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))  # Adaptive padding
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

# Function to create activation layers based on the type specified
def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Creates an activation layer of specified type: ReLU, LeakyReLU, or PReLU.
    These activations introduce non-linearity into the network, allowing it to learn complex patterns.

    Parameters:
    - act_type: Type of activation function ('relu', 'lrelu', or 'prelu').
    - inplace: Boolean flag to modify the input in-place for memory efficiency.
    - neg_slope: Negative slope for LeakyReLU and PReLU, controlling how much the activation allows negative values.
    - n_prelu: Number of learned parameters for PReLU.

    Returns:
    - The activation layer based on the specified type.
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(f'Activation layer [{act_type}] not found')
    return layer

# Function to sequentially stack layers/modules
def sequential(*args):
    """
    Creates a sequential container that stacks modules in the order they are passed.
    Useful for building complex models by stacking layers in a pipeline fashion.

    Parameters:
    - args: List of layers or modules to be stacked sequentially.

    Returns:
    - A Sequential container with the specified modules.
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('Sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

# PixelShuffle-based upsampling block
def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3):
    """
    Upsampling block that increases the spatial resolution of the input.
    Uses PixelShuffle, which rearranges elements to form higher-resolution outputs.

    Parameters:
    - in_channels: Number of input channels.
    - out_channels: Number of output channels.
    - upscale_factor: Factor by which resolution is increased.
    - kernel_size: Size of the convolutional kernel.

    Returns:
    - A sequential container with a convolution and a pixel shuffle operation.
    """
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)  # Rearranges feature map dimensions
    return sequential(conv, pixel_shuffle)

# Enhanced Spatial Attention (ESA) Block
class ESA(nn.Module):
    """
    Enhanced Spatial Attention (ESA) block to enhance important spatial features.
    This module uses several convolutions and pooling layers to focus on significant regions.

    Parameters:
    - esa_channels: Number of channels for ESA processing.
    - n_feats: Number of input features.
    - conv: Convolution layer function (e.g., nn.Conv2d).
    """
    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()  # Activation to create a mask
        self.relu = nn.ReLU(inplace=True)  # Activation function

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)  # Down-sample important areas
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)  # Upsample back
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)  # Combine features
        m = self.sigmoid(c4)  # Mask
        return x * m  # Apply mask to original input

# Residual Local Feature Block (RLFB)
class RLFB(nn.Module):
    """
    Residual Local Feature Block for enhancing local features and maintaining information with residual connections.
    
    Parameters:
    - in_channels: Number of input channels.
    - mid_channels: Number of channels in intermediate layers.
    - out_channels: Number of output channels.
    - esa_channels: Channels for the ESA block.
    """
    def __init__(self, in_channels, mid_channels=None, out_channels=None, esa_channels=16):
        super(RLFB, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = conv_layer(in_channels, mid_channels, 3)  # First convolution layer
        self.c2_r = conv_layer(mid_channels, mid_channels, 3)  # Second convolution layer
        self.c3_r = conv_layer(mid_channels, in_channels, 3)  # Third convolution layer
        self.c5 = conv_layer(in_channels, out_channels, 1)  # Final layer before ESA
        self.esa = ESA(esa_channels, out_channels, nn.Conv2d)  # ESA block for spatial attention
        self.act = activation('lrelu', neg_slope=0.05)  # Leaky ReLU activation

    def forward(self, x):
        out = self.c1_r(x)
        out = self.act(out)
        out = self.c2_r(out)
        out = self.act(out)
        out = self.c3_r(out)
        out = self.act(out)
        out = out + x  # Residual connection, maintaining information from input
        out = self.esa(self.c5(out))  # Enhanced Spatial Attention
        return out

# Residual Local Feature Network (RLFN)
class RLFN_Prune(nn.Module):
    """
    Main model combining multiple RLFB blocks for image super-resolution.

    Parameters:
    - in_channels: Number of input channels (e.g., RGB = 3).
    - out_channels: Number of output channels.
    - feature_channels: Number of feature channels in the initial conv layer.
    - mid_channels: Intermediate channels in RLFB blocks.
    - upscale: Upscaling factor for the final output.
    """
    def __init__(self, in_channels=3, out_channels=3, feature_channels=46, mid_channels=48, upscale=4):
        super(RLFN_Prune, self).__init__()
        self.conv_1 = conv_layer(in_channels, feature_channels, kernel_size=3)  # Initial conv layer
        self.block_1 = RLFB(feature_channels, mid_channels)  # First RLFB block
        self.block_2 = RLFB(feature_channels, mid_channels)  # Second RLFB block
        self.block_3 = RLFB(feature_channels, mid_channels)  # Third RLFB block
        self.block_4 = RLFB(feature_channels, mid_channels)  # Fourth RLFB block
        self.conv_2 = conv_layer(feature_channels, feature_channels, kernel_size=3)  # Intermediate conv layer
        self.upsampler = pixelshuffle_block(feature_channels, out_channels, upscale_factor=upscale)  # Upsampling block

    def forward(self, x):
        out_feature = self.conv_1(x)  # Initial features
        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)
        out_low_resolution = self.conv_2(out_b4) + out_feature  # Residual connection with initial feature map
        output = self.upsampler(out_low_resolution)  # Upsample to the target resolution
        return output
