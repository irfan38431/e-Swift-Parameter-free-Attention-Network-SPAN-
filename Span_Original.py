from collections import OrderedDict
import torch
from torch import nn as nn
import torch.nn.functional as F

# Define a custom convolutional layer class with configurable options
class Conv3XC2(nn.Module):
    def __init__(self, c_in, c_out, gain1=1, gain2=0, s=1, groups=1, bias=True, relu=False):
        super(Conv3XC2, self).__init__()  # Initialize parent class
        self.weight_concat = None         # Placeholder for concatenated weights
        self.bias_concat = None           # Placeholder for concatenated biases
        self.update_params_flag = False   # Flag to control weight update
        self.stride = s                   # Set the stride value
        self.has_relu = relu              # Check if ReLU activation is needed
        self.groups = groups              # Number of groups for grouped convolution
        self.gain = gain1                 # Gain multiplier for number of channels
        gain = gain1                      # Assign gain multiplier

        # Define a 1x1 convolution layer (shortcut layer)
        self.sk = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=1,
            padding=0,
            stride=s,
            bias=bias,
            groups=groups,
        )

        # Define a sequence of convolutional layers for processing input
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_in * gain,
                kernel_size=1,
                padding=0,
                bias=bias,
                groups=groups,
            ),
            nn.Conv2d(
                in_channels=c_in * gain,
                out_channels=c_out * gain,
                kernel_size=3,
                stride=s,
                padding=0,
                bias=bias,
                groups=groups,
            ),
            nn.Conv2d(
                in_channels=c_out * gain,
                out_channels=c_out,
                kernel_size=1,
                padding=0,
                bias=bias,
                groups=groups,
            ),
        )

        # Define an evaluation convolution layer to use concatenated weights
        self.eval_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            padding=1,
            stride=s,
            bias=bias,
            groups=groups,
        )
        self.eval_conv.weight.requires_grad = False  # Freeze weights in eval mode
        self.eval_conv.bias.requires_grad = False    # Freeze biases in eval mode
        self.update_params()                         # Initialize parameters

    # Update parameters function to concatenate weights and biases for evaluation
    def update_params(self):
        # Perform grouped convolution weight update
        with torch.no_grad():  # Disable gradient computation
            # Clone each convolution layer's weights and biases
            w1 = self.conv[0].weight.data.clone()
            b1 = self.conv[0].bias.data.clone() if self.conv[0].bias is not None else None
            w2 = self.conv[1].weight.data.clone()
            b2 = self.conv[1].bias.data.clone() if self.conv[1].bias is not None else None
            w3 = self.conv[2].weight.data.clone()
            b3 = self.conv[2].bias.data.clone() if self.conv[2].bias is not None else None

            # Grouped convolution: perform operations separately for each group
            group_in_channels = w1.size(1) // self.groups       # Input channels per group
            group_out_channels = w3.size(0) // self.groups      # Output channels per group

            # Initialize zero tensors for concatenated weights and biases
            weight_concat = torch.zeros_like(self.eval_conv.weight.data)
            bias_concat = torch.zeros_like(self.eval_conv.bias.data) if self.eval_conv.bias is not None else None

            # Process each group separately
            for g in range(self.groups):
                # Extract weights for each group from convolution layers
                w1_g = w1[g * group_out_channels * self.gain : (g + 1) * group_out_channels * self.gain, :, :, :]
                w2_g = w2[g * group_out_channels * self.gain : (g + 1) * group_out_channels * self.gain, :, :, :]
                w3_g = w3[g * group_out_channels : (g + 1) * group_out_channels, :, :, :]

                # Perform convolution on grouped weights
                w_g = F.conv2d(w1_g.flip(2, 3).permute(1, 0, 2, 3), w2_g, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)

                # Compute final weight for each group by applying the third convolution
                weight_concat_g = F.conv2d(w_g.flip(2, 3).permute(1, 0, 2, 3), w3_g, padding=0, stride=1).flip(2, 3).permute(1, 0, 2, 3)

                # Store concatenated weights for the current group
                weight_concat[g * group_out_channels : (g + 1) * group_out_channels, :, :, :] = weight_concat_g

                # If there are biases, combine them as well
                if b1 is not None and b2 is not None and b3 is not None:
                    b_g = (w2_g * b1[g * group_out_channels * self.gain : (g + 1) * group_out_channels * self.gain].reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2[g * group_out_channels * self.gain : (g + 1) * group_out_channels * self.gain]
                    bias_concat_g = (w3_g * b_g.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3[g * group_out_channels : (g + 1) * group_out_channels]
                    bias_concat[g * group_out_channels : (g + 1) * group_out_channels] = bias_concat_g

            # Add shortcut layer's weights and biases if present
            sk_w = self.sk.weight.data.clone()
            sk_b = self.sk.bias.data.clone() if self.sk.bias is not None else None
            if sk_w is not None:
                H_pixels_to_pad = (self.eval_conv.kernel_size[0] - 1) // 2
                W_pixels_to_pad = (self.eval_conv.kernel_size[1] - 1) // 2
                sk_w_padded = F.pad(sk_w, [W_pixels_to_pad, W_pixels_to_pad, H_pixels_to_pad, H_pixels_to_pad])

                # Add padded weights to concatenated weights
                weight_concat += sk_w_padded

            # Add shortcut biases if they exist
            if sk_b is not None:
                bias_concat += sk_b

            # Update the eval_conv layer with concatenated weights and biases
            self.eval_conv.weight.data = weight_concat
            if self.eval_conv.bias is not None:
                self.eval_conv.bias.data = bias_concat

    # Define forward pass
    def forward(self, x):
        if self.training:  # During training, use the full convolution sequence
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)  # Combine convolution and shortcut layer
        else:
            self.update_params()  # Update parameters for evaluation
            out = self.eval_conv(x)  # Use eval_conv in eval mode

        if self.has_relu:  # Apply ReLU activation if specified
            out = F.leaky_relu(out, negative_slope=0.05)
        return out


# Utility function to ensure kernel size is in (x, y) pair format
def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2  # Convert single integer to tuple (x, x)
    return value

# Custom convolution layer that shifts channels in up, down, left, right directions
class ShiftConv2d_4(nn.Module):
    def __init__(self, inp_channels, move_channels=2, move_pixels=1):
        super(ShiftConv2d_4, self).__init__()
        self.inp_channels = inp_channels  # Set input channels
        self.move_p = move_pixels         # Number of pixels to move
        self.move_c = move_channels       # Number of channels to move in each direction
        # Non-trainable weight initialized with zeros for shifting
        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)

        # Define channel range for each shift direction
        mid_channel = inp_channels // 2
        up_channels = (mid_channel - move_channels * 2, mid_channel - move_channels)
        down_channels = (mid_channel - move_channels, mid_channel)
        left_channels = (mid_channel, mid_channel + move_channels)
        right_channels = (mid_channel + move_channels, mid_channel + move_channels * 2)

        # Set weights to shift in each direction: left, right, up, down
        self.weight[left_channels[0] : left_channels[1], 0, 1, 2] = 1.0  # left shift
        self.weight[right_channels[0] : right_channels[1], 0, 1, 0] = 1.0  # right shift
        self.weight[up_channels[0] : up_channels[1], 0, 2, 1] = 1.0  # up shift
        self.weight[down_channels[0] : down_channels[1], 0, 0, 1] = 1.0  # down shift
        self.weight[0 : mid_channel - move_channels * 2, 0, 1, 1] = 1.0  # identity
        self.weight[mid_channel + move_channels * 2 :, 0, 1, 1] = 1.0  # identity

    # Forward pass to apply the shift using convolution
    def forward(self, x):
        for i in range(self.move_p):  # Repeat shifting move_p times
            # Apply convolution for shifting effect with pre-defined weights
            x = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.inp_channels)
        return x


# Define a block combining pointwise (1x1) and depthwise convolutions
class BSConvU(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode="zeros", with_bn=False, bn_kwargs=None):
        super().__init__()  # Initialize the parent class
        
        # Default arguments for batch normalization if not provided
        if bn_kwargs is None:
            bn_kwargs = {}

        # Pointwise convolution (1x1) to adjust the number of channels
        self.add_module("pw", torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        ))

        # Add batch normalization if requested
        if with_bn:
            self.add_module("bn", torch.nn.BatchNorm2d(num_features=out_channels, **bn_kwargs))

        # Depthwise convolution, applying convolution individually on each channel
        self.add_module("dw", torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,  # Depthwise operation
            bias=bias,
            padding_mode=padding_mode,
        ))

# Helper function to create a convolutional layer with adaptive padding
def conv_layer(in_channels, out_channels, kernel_size, bias=True):
    kernel_size = _make_pair(kernel_size)  # Ensure kernel size is a tuple
    # Calculate padding to keep input and output sizes the same
    padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

# Function to select and return activation function based on string input
def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].
    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()  # Convert activation type to lowercase
    if act_type == "relu":
        layer = nn.ReLU(inplace)  # ReLU activation
    elif act_type == "lrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)  # Leaky ReLU with specified negative slope
    elif act_type == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)  # PReLU with learnable slope
    else:
        # If unknown activation type is provided, raise an error
        raise NotImplementedError(f"activation layer [{act_type}] is not found")
    return layer

# Sequentially stack layers in the order they are passed as arguments
def sequential(*args):
    if len(args) == 1 and isinstance(args[0], OrderedDict):
        # Raise error if OrderedDict is passed, as it is not supported here
        raise NotImplementedError("sequential does not support OrderedDict input.")
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):  # If module is Sequential, add its submodules
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):  # If it’s a module, add it directly
            modules.append(module)
    return nn.Sequential(*modules)

# Create a pixel shuffle block for upsampling
def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3):
    """
    Upsample features according to `upscale_factor`.
    """
    # Initial convolution layer with channels multiplied by upscale factor squared
    conv = conv_layer(in_channels, out_channels * (upscale_factor**2), kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)  # Pixel shuffle for upsampling
    return sequential(conv, pixel_shuffle)  # Return the sequential block

# Another custom convolution layer class with specific configuration
class Conv3XC(nn.Module):
    def __init__(self, c_in, c_out, gain1=1, gain2=0, s=1, bias=True, relu=False):
        super(Conv3XC, self).__init__()  # Initialize parent class
        self.weight_concat = None         # Placeholder for concatenated weights
        self.bias_concat = None           # Placeholder for concatenated biases
        self.update_params_flag = False   # Flag for updating parameters
        self.stride = s                   # Set the stride value
        self.has_relu = relu              # Check if ReLU activation is needed
        gain = gain1                      # Assign gain multiplier

        # Define a convolution layer for evaluation with padding to maintain size
        self.eval_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            padding=1,
            stride=s,
            bias=bias,
        )
        self.eval_conv.weight.requires_grad = False  # Freeze weights in eval mode
        self.eval_conv.bias.requires_grad = False    # Freeze biases in eval mode

    # Function to update parameters (this section is commented out here)
    def update_params(self):
        w1 = self.conv[0].weight.data.clone().detach()  # Clone weights of first layer
        b1 = self.conv[0].bias.data.clone().detach()    # Clone biases of first layer
        w2 = self.conv[1].weight.data.clone().detach()  # Clone weights of second layer
        b2 = self.conv[1].bias.data.clone().detach()    # Clone biases of second layer
        w3 = self.conv[2].weight.data.clone().detach()  # Clone weights of third layer
        b3 = self.conv[2].bias.data.clone().detach()    # Clone biases of third layer

        # Convolution operations on the weights to compute concatenated weights and biases
        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        # Final concatenation of weights
        self.weight_concat = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        # Additional padding to match kernel size in eval_conv layer
        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()
        target_kernel_size = 3
        H_pixels_to_pad = (target_kernel_size - 1) // 2
        W_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(sk_w, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])

        self.weight_concat += sk_w  # Update concatenated weights
        self.bias_concat += sk_b    # Update concatenated biases

        # Update eval_conv layer with concatenated weights and biases
        self.eval_conv.weight.data = self.weight_concat
        self.eval_conv.bias.data = self.bias_concat

    # Define forward pass through the model
    def forward(self, x):
        out = self.eval_conv(x)  # Apply eval_conv layer
        if self.has_relu:  # Apply ReLU if specified
            out = F.leaky_relu(out, negative_slope=0.05)
        return out



# Define a custom activation function class
class CustomActivation(nn.Module):
    def __init__(self, num_channels):
        super(CustomActivation, self).__init__()  # Initialize the parent class
        # Define a learnable parameter alpha for each channel
        self.alpha = nn.Parameter(torch.ones((1, num_channels, 1, 1)), requires_grad=True)

    # Forward pass applies sigmoid activation scaled by alpha
    def forward(self, x):
        return x * torch.sigmoid(self.alpha * x)

# Define a slim block for lightweight convolution and custom activation
class SlimBlock(nn.Module):
    def __init__(self, c):
        super().__init__()  # Initialize the parent class
        # Set the number of channels for depthwise convolution
        dw_channel = c
        # Depthwise convolution layer
        self.conv1 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=2, bias=True)
        self.act = CustomActivation(c)  # Custom activation function

    # Forward pass through depthwise convolution and activation
    def forward(self, inp):
        x = self.conv1(inp)  # Apply depthwise convolution
        x = self.act(x)      # Apply custom activation
        y = x + inp          # Add input to the result (residual connection)
        return y

# Define SPAB1 block: multi-pathway block with custom convolutions and attention
class SPAB1(nn.Module):
    def __init__(self, in_channels, mid_channels=None, out_channels=None, bias=False):
        super(SPAB1, self).__init__()  # Initialize the parent class
        mid_channels = mid_channels or in_channels  # Set mid_channels to in_channels if not specified
        out_channels = out_channels or in_channels  # Set out_channels to in_channels if not specified

        # Define a sequence of convolutions for main pathway
        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1)

        # Activation function for pathway
        self.act1 = torch.nn.SiLU(inplace=True)

    # Forward pass that applies convolutions and combines with input through attention
    def forward(self, x):
        out1 = self.c1_r(x)               # First convolution
        out1_act = self.act1(out1)        # Apply activation
        out2 = self.c2_r(out1_act)        # Second convolution
        out2_act = self.act1(out2)        # Apply activation
        out3 = self.c3_r(out2_act)        # Third convolution

        # Apply attention mechanism
        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att  # Scale by attention and combine with input
        return out, out1, sim_att

# Define SPAB2 block: similar to SPAB1 but uses grouped convolutions
class SPAB2(nn.Module):
    def __init__(self, in_channels, mid_channels=None, out_channels=None, bias=False):
        super(SPAB2, self).__init__()  # Initialize parent class
        mid_channels = mid_channels or in_channels
        out_channels = out_channels or in_channels

        # Define a sequence of grouped convolutions for main pathway
        self.c1_r = Conv3XC2(in_channels, mid_channels, gain1=2, s=1, groups=2)
        self.c2_r = Conv3XC2(mid_channels, mid_channels, gain1=2, s=1, groups=2)
        self.c3_r = Conv3XC2(mid_channels, out_channels, gain1=2, s=1, groups=2)

        # Define custom activations for the pathways
        self.act1 = CustomActivation(mid_channels)
        self.act2 = CustomActivation(mid_channels)
        self.act = torch.nn.SiLU(inplace=True)

    # Forward pass through convolutions and attention mechanism
    def forward(self, x):
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        out2_act = self.act2(out2)

        out3 = self.c3_r(out2_act)
        out3 = self.act(out3) + x  # Combine with input

        return out3, out1, out3

# Define the main super-resolution model: SPAN30
class SPAN30(nn.Module):
    """
    Swift Parameter-free Attention Network for Efficient Super-Resolution
    """
    def __init__(self, num_in_ch, num_out_ch, feature_channels=48, upscale=4, bias=True, img_range=255.0, rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(SPAN30, self).__init__()  # Initialize parent class

        # Define input and output channels, and set image normalization parameters
        in_channels = num_in_ch
        out_channels = num_out_ch
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)  # Mean values for RGB normalization

        # Initial convolution layer
        self.conv_1 = Conv3XC(in_channels, feature_channels, gain1=2, s=1)

        # Define a series of SPAB1 blocks for feature extraction
        self.block_1 = SPAB1(feature_channels, bias=bias)
        self.block_2 = SPAB1(feature_channels, bias=bias)
        self.block_3 = SPAB1(feature_channels, bias=bias)
        self.block_4 = SPAB1(feature_channels, bias=bias)
        self.block_5 = SPAB1(feature_channels, bias=bias)
        self.block_6 = SPAB1(feature_channels, bias=bias)

        # Concatenation layer and final convolution for reducing dimensions
        self.conv_cat = conv_layer(feature_channels * 4, feature_channels, kernel_size=1, bias=True)
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain1=2, s=1)

        # Upsampling layer for increasing resolution
        self.upsampler = pixelshuffle_block(feature_channels, out_channels, upscale_factor=upscale)
        self.cuda()(torch.randn(1, 3, 256, 256).cuda())  # Preload tensor to GPU for initialization

    # Forward pass through SPAN30 model
    def forward(self, x):
        self.mean = self.mean.type_as(x)  # Ensure mean tensor has the same type as input
        x = (x - self.mean) * self.img_range  # Normalize input image
        out_feature = self.conv_1(x)  # Initial convolution

        # Sequentially pass through SPAB1 blocks for feature extraction
        out_b1, out_b0_2, att1 = self.block_1(out_feature)
        out_b2, out_b1_2, att2 = self.block_2(out_b1)
        out_b3, out_b2_2, att3 = self.block_3(out_b2)
        out_b4, out_b3_2, att4 = self.block_4(out_b3)
        out_b5, out_b4_2, att5 = self.block_5(out_b4)
        out_b6, out_b5_2, att6 = self.block_6(out_b5)

        out_final = self.conv_2(out_b6)  # Final convolution on extracted features
        # Concatenate outputs and reduce dimensions
        out = self.conv_cat(torch.cat([out_feature, out_final, out_b1, out_b5_2], 1))
        output = self.upsampler(out)  # Upsample the final output for super-resolution
        return output

# Main section to test and evaluate the model's performance
if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table  # Import FLOP analysis tools
    import time

    # Instantiate SPAN30 model, set to evaluation mode, and prepare random input
    model = SPAN30(3, 3, upscale=4, feature_channels=48).cuda()
    model.eval()  # Set model to evaluation mode
    inputs = (torch.rand(1, 3, 256, 256).cuda(),)  # Create random input tensor for testing
    print(flop_count_table(FlopCountAnalysis(model, inputs)))  # Print model FLOPs

    # Measure runtime of the model by averaging over multiple passes
    total_time = 0
    input_x = torch.rand(1, 3, 512, 512).cuda()  # Larger input tensor for runtime test
    for i in range(100):  # Run model 100 times for averaging timing
        torch.cuda.empty_cache()  # Clear CUDA cache to prevent memory overflow
        sta_time = time.time()    # Start time measurement
        model(input_x)            # Run forward pass
        one_time = time.time() - sta_time  # Calculate time taken for this pass
        total_time += one_time * 1000      # Convert to milliseconds and add to total
        print("idx: {} one time: {:.4f} ms".format(i, one_time))  # Print each pass time
    print("Avg time: {:.4f}".format(total_time / 100.0))  # Print average runtime across all passes

