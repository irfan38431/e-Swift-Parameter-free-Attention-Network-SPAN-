from collections import OrderedDict
import torch
from torch import nn as nn
import torch.nn.functional as F

# Define a custom convolutional layer class with specific features
class Conv3XC2(nn.Module):
    def __init__(self, c_in, c_out, gain1=1, gain2=0, s=1, groups=1, bias=True, relu=False):
        super(Conv3XC2, self).__init__()  # Initialize parent class
        self.weight_concat = None         # Store combined weights for convolution layers
        self.bias_concat = None           # Store combined biases
        self.update_params_flag = False   # Flag for updating parameters
        self.stride = s                   # Set stride
        self.has_relu = relu              # Check if ReLU is required
        self.groups = groups              # Set groups for group convolutions
        self.gain = gain1                 # Gain multiplier for channels
        gain = gain1                      # Another variable for gain

        # Define 1x1 convolution layer with set parameters
        self.sk = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1, padding=0, stride=s, bias=bias, groups=groups)
        
        # Define a sequence of convolutional layers with specified gains
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_in * gain, kernel_size=1, padding=0, bias=bias, groups=groups),
            nn.Conv2d(in_channels=c_in * gain, out_channels=c_out * gain, kernel_size=3, stride=s, padding=0, bias=bias, groups=groups),
            nn.Conv2d(in_channels=c_out * gain, out_channels=c_out, kernel_size=1, padding=0, bias=bias, groups=groups),
        )

        # Define another convolution layer for evaluation mode
        self.eval_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=s, bias=bias, groups=groups)
        self.eval_conv.weight.requires_grad = False  # Freeze weights in eval mode
        self.eval_conv.bias.requires_grad = False    # Freeze bias in eval mode
        self.update_params()                         # Initialize parameters

    # Function to update parameters in eval mode
    def update_params(self):
        with torch.no_grad():  # Disable gradient tracking
            # Clone weights and biases for each layer
            w1 = self.conv[0].weight.data.clone()
            b1 = self.conv[0].bias.data.clone() if self.conv[0].bias is not None else None
            w2 = self.conv[1].weight.data.clone()
            b2 = self.conv[1].bias.data.clone() if self.conv[1].bias is not None else None
            w3 = self.conv[2].weight.data.clone()
            b3 = self.conv[2].bias.data.clone() if self.conv[2].bias is not None else None
            
            # Calculate channels per group
            group_in_channels = w1.size(1) // self.groups
            group_out_channels = w3.size(0) // self.groups

            # Initialize zero-tensors for concatenated weights and biases
            weight_concat = torch.zeros_like(self.eval_conv.weight.data)
            bias_concat = torch.zeros_like(self.eval_conv.bias.data) if self.eval_conv.bias is not None else None

            # Loop through each group and perform weight and bias concatenations
            for g in range(self.groups):
                # Split and flip weights for each layer within the group
                w1_g = w1[g * group_out_channels * self.gain : (g + 1) * group_out_channels * self.gain, :, :, :]
                w2_g = w2[g * group_out_channels * self.gain : (g + 1) * group_out_channels * self.gain, :, :, :]
                w3_g = w3[g * group_out_channels : (g + 1) * group_out_channels, :, :, :]

                # Perform convolution on split weights
                w_g = F.conv2d(w1_g.flip(2, 3).permute(1, 0, 2, 3), w2_g, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)

                # Perform second convolution and concatenate weights
                weight_concat_g = F.conv2d(w_g.flip(2, 3).permute(1, 0, 2, 3), w3_g, padding=0, stride=1).flip(2, 3).permute(1, 0, 2, 3)
                weight_concat[g * group_out_channels : (g + 1) * group_out_channels, :, :, :] = weight_concat_g

                # Concatenate biases if available
                if b1 is not None and b2 is not None and b3 is not None:
                    b_g = (w2_g * b1[g * group_out_channels * self.gain : (g + 1) * group_out_channels * self.gain].reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2[g * group_out_channels * self.gain : (g + 1) * group_out_channels * self.gain]
                    bias_concat_g = (w3_g * b_g.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3[g * group_out_channels : (g + 1) * group_out_channels]
                    bias_concat[g * group_out_channels : (g + 1) * group_out_channels] = bias_concat_g

            # Pad sk layer weights and biases
            sk_w = self.sk.weight.data.clone()
            sk_b = self.sk.bias.data.clone() if self.sk.bias is not None else None
            if sk_w is not None:
                H_pixels_to_pad = (self.eval_conv.kernel_size[0] - 1) // 2
                W_pixels_to_pad = (self.eval_conv.kernel_size[1] - 1) // 2
                sk_w_padded = F.pad(sk_w, [W_pixels_to_pad, W_pixels_to_pad, H_pixels_to_pad, H_pixels_to_pad])

                # Add padded weights to concatenated weights
                weight_concat += sk_w_padded

            # Add sk biases if present
            if sk_b is not None:
                bias_concat += sk_b

            # Update eval_conv weights and biases
            self.eval_conv.weight.data = weight_concat
            if self.eval_conv.bias is not None:
                self.eval_conv.bias.data = bias_concat

    # Define forward pass for the model
    def forward(self, x):
        # In training mode, apply padding and calculate output
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:  # In eval mode, use precomputed parameters
            self.update_params()
            out = self.eval_conv(x)

        # Apply ReLU activation if specified
        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out


# Utility function to handle both single integer and tuple input for kernel size
def _make_pair(value):
    if isinstance(value, int):  # Check if the input is an integer
        value = (value,) * 2    # If integer, convert to tuple of two identical values
    return value

# Custom layer that shifts channels up, down, left, or right
class ShiftConv2d_4(nn.Module):
    def __init__(self, inp_channels, move_channels=2, move_pixels=1):
        super(ShiftConv2d_4, self).__init__()  # Initialize parent class
        self.inp_channels = inp_channels       # Number of input channels
        self.move_p = move_pixels              # Number of pixels to shift
        self.move_c = move_channels            # Number of channels to move
        # Create a non-trainable weight parameter initialized with zeros
        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)
        
        # Define channel ranges for each direction shift
        mid_channel = inp_channels // 2
        up_channels = (mid_channel - move_channels * 2, mid_channel - move_channels)
        down_channels = (mid_channel - move_channels, mid_channel)
        left_channels = (mid_channel, mid_channel + move_channels)
        right_channels = (mid_channel + move_channels, mid_channel + move_channels * 2)

        # Set weights for each direction shift in the weight tensor
        self.weight[left_channels[0] : left_channels[1], 0, 1, 2] = 1.0  # Shift left
        self.weight[right_channels[0] : right_channels[1], 0, 1, 0] = 1.0  # Shift right
        self.weight[up_channels[0] : up_channels[1], 0, 2, 1] = 1.0  # Shift up
        self.weight[down_channels[0] : down_channels[1], 0, 0, 1] = 1.0  # Shift down
        self.weight[0 : mid_channel - move_channels * 2, 0, 1, 1] = 1.0  # Identity
        self.weight[mid_channel + move_channels * 2 :, 0, 1, 1] = 1.0  # Identity

    # Forward pass with shifting using convolution
    def forward(self, x):
        for i in range(self.move_p):  # Repeat shifting as specified by move_p
            # Perform convolution to shift channels as per defined weights
            x = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.inp_channels)
        return x

# BSConvU block using pointwise (1x1) and depthwise (grouped) convolutions
class BSConvU(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode="zeros", with_bn=False, bn_kwargs=None):
        super().__init__()  # Initialize parent class
        
        if bn_kwargs is None:  # Set batch norm parameters if not provided
            bn_kwargs = {}

        # Pointwise convolution (1x1) to adjust channels
        self.add_module("pw", torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1, padding=0, dilation=1, groups=1, bias=False))

        if with_bn:  # Add BatchNorm layer if specified
            self.add_module("bn", torch.nn.BatchNorm2d(num_features=out_channels, **bn_kwargs))

        # Depthwise convolution with specified kernel size and other parameters
        self.add_module("dw", torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=out_channels, bias=bias, padding_mode=padding_mode))

# Create convolution layer with adaptive padding
def conv_layer(in_channels, out_channels, kernel_size, bias=True):
    kernel_size = _make_pair(kernel_size)  # Ensure kernel size is a tuple
    padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))  # Calculate padding
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

# Activation function selection function
def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()  # Ensure lowercase input for act_type
    if act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type == "lrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError("activation layer [{:s}] is not found".format(act_type))
    return layer

# Create a sequential container for a series of modules
def sequential(*args):
    if len(args) == 1 and isinstance(args[0], OrderedDict):
        raise NotImplementedError("sequential does not support OrderedDict input.")
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

# Upsampling layer using Pixel Shuffle
def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3):
    conv = conv_layer(in_channels, out_channels * (upscale_factor**2), kernel_size)  # Initial convolution layer
    pixel_shuffle = nn.PixelShuffle(upscale_factor)  # Pixel shuffle for upsampling
    return sequential(conv, pixel_shuffle)

# Custom convolutional layer class with specific gains and options for ReLU
class Conv3XC(nn.Module):
    def __init__(self, c_in, c_out, gain1=1, gain2=0, s=1, bias=True, relu=False):
        super(Conv3XC, self).__init__()  # Initialize parent class
        self.weight_concat = None         # Store combined weights
        self.bias_concat = None           # Store combined biases
        self.update_params_flag = False   # Flag for updating parameters
        self.stride = s                   # Set stride
        self.has_relu = relu              # Check if ReLU is needed

        # Define a convolutional layer for evaluation
        self.eval_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=s, bias=bias)
        self.eval_conv.weight.requires_grad = False  # Freeze weights in eval mode
        self.eval_conv.bias.requires_grad = False    # Freeze bias in eval mode

    # Function to update parameters in eval mode
    def update_params(self):
        # Copy weights and biases, performing convolutions to update them
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        self.weight_concat = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()
        target_kernel_size = 3

        H_pixels_to_pad = (target_kernel_size - 1) // 2
        W_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(sk_w, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])

        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b

        self.eval_conv.weight.data = self.weight_concat
        self.eval_conv.bias.data = self.bias_concat

    # Forward pass through the layer with optional ReLU activation
    def forward(self, x):
        out = self.eval_conv(x)
        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out



# Custom activation function using a learnable parameter
class CustomActivation(nn.Module):
    def __init__(self, num_channels):
        super(CustomActivation, self).__init__()  # Initialize parent class
        # Initialize alpha parameter for each channel
        self.alpha = nn.Parameter(torch.ones((1, num_channels, 1, 1)), requires_grad=True)

    # Forward pass where input is scaled by sigmoid of alpha
    def forward(self, x):
        return x * torch.sigmoid(self.alpha * x)

# SlimBlock defines a lightweight block with depthwise convolution and custom activation
class SlimBlock(nn.Module):
    def __init__(self, c):
        super().__init__()  # Initialize parent class
        dw_channel = c  # Set number of depthwise channels
        # Define depthwise convolution with specified channel count
        self.conv1 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=2, bias=True)
        self.act = CustomActivation(c)  # Custom activation for this block

    # Forward pass combining input with depthwise convolution and activation
    def forward(self, inp):
        x = self.conv1(inp)  # Apply depthwise convolution
        x = self.act(x)      # Apply custom activation
        y = x + inp          # Residual connection (add input to output)
        return y

# SPAB1 block - a multi-pathway block with custom convolutions and attention mechanism
class SPAB1(nn.Module):
    def __init__(self, in_channels, mid_channels=None, out_channels=None, bias=False):
        super(SPAB1, self).__init__()  # Initialize parent class
        mid_channels = mid_channels or in_channels  # Default mid_channels to in_channels if None
        out_channels = out_channels or in_channels  # Default out_channels to in_channels if None

        # Define the main convolution pathway with three convolution layers
        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1)

        # Define an additional parallel convolution pathway with three layers
        self.extra_c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1)
        self.extra_c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1)
        self.extra_c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1)

        # Activation function applied between convolutions
        self.act1 = torch.nn.SiLU(inplace=True)

    # Forward pass that combines two pathways and applies attention
    def forward(self, x):
        # First pathway through three convolutions and activations
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)
        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)
        out3 = self.c3_r(out2_act)

        # Second parallel pathway through three convolutions and activations
        extra_out1 = self.extra_c1_r(x)
        extra_out1_act = self.act1(extra_out1)
        extra_out2 = self.extra_c2_r(extra_out1_act)
        extra_out2_act = self.act1(extra_out2)
        extra_out3 = self.extra_c3_r(extra_out2_act)

        # Combine outputs from both pathways
        combined_out = out3 + extra_out3

        # Apply a simple attention mechanism to the combined output
        sim_att = torch.sigmoid(combined_out) - 0.5
        out = (combined_out + x) * sim_att  # Apply attention scaling
        return out, out1, sim_att

# SPAB2 block - similar to SPAB1, but uses Conv3XC2 for grouped convolutions
class SPAB2(nn.Module):
    def __init__(self, in_channels, mid_channels=None, out_channels=None, bias=False):
        super(SPAB2, self).__init__()  # Initialize parent class
        mid_channels = mid_channels or in_channels  # Default mid_channels to in_channels if None
        out_channels = out_channels or in_channels  # Default out_channels to in_channels if None

        # Define main convolution pathway using Conv3XC2
        self.c1_r = Conv3XC2(in_channels, mid_channels, gain1=2, s=1, groups=2)
        self.c2_r = Conv3XC2(mid_channels, mid_channels, gain1=2, s=1, groups=2)
        self.c3_r = Conv3XC2(mid_channels, out_channels, gain1=2, s=1, groups=2)

        # Define parallel convolution pathway using Conv3XC2
        self.extra_c1_r = Conv3XC2(in_channels, mid_channels, gain1=2, s=1, groups=2)
        self.extra_c2_r = Conv3XC2(mid_channels, mid_channels, gain1=2, s=1, groups=2)
        self.extra_c3_r = Conv3XC2(mid_channels, out_channels, gain1=2, s=1, groups=2)

        # Custom activation for each pathway
        self.act1 = CustomActivation(mid_channels)
        self.act2 = CustomActivation(mid_channels)
        self.act = torch.nn.SiLU(inplace=True)

    # Forward pass that combines pathways and applies custom activations
    def forward(self, x):
        # Original pathway
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)
        out2 = self.c2_r(out1_act)
        out2_act = self.act2(out2)
        out3 = self.c3_r(out2_act)

        # Parallel pathway
        extra_out1 = self.extra_c1_r(x)
        extra_out1_act = self.act1(extra_out1)
        extra_out2 = self.extra_c2_r(extra_out1_act)
        extra_out2_act = self.act2(extra_out2)
        extra_out3 = self.extra_c3_r(extra_out2_act)

        # Combine outputs from both pathways and apply final activation
        combined_out = out3 + extra_out3
        out3 = self.act(combined_out) + x  # Residual connection with input
        return out3, out1, out3

# SPAN30 class, the main model, implementing a super-resolution network
class SPAN30(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, feature_channels=48, upscale=4, bias=True, img_range=255.0, rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(SPAN30, self).__init__()  # Initialize parent class
        self.img_range = img_range  # Set image range (255 for 8-bit images)
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)  # Image mean for normalization

        # Initial convolution layer
        self.conv_1 = Conv3XC(num_in_ch, feature_channels, gain1=2, s=1)

        # Stack of attention-based blocks for feature extraction
        self.block_1 = SPAB1(feature_channels, bias=bias)
        self.block_2 = SPAB1(feature_channels, bias=bias)
        self.block_3 = SPAB1(feature_channels, bias=bias)
        self.block_4 = SPAB1(feature_channels, bias=bias)
        self.block_5 = SPAB1(feature_channels, bias=bias)
        self.block_6 = SPAB1(feature_channels, bias=bias)

        # Concatenate features, reducing dimensions with a 1x1 convolution
        self.conv_cat = conv_layer(feature_channels * 4, feature_channels, kernel_size=1, bias=True)
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain1=2, s=1)

        # Upsampling layer to increase resolution
        self.upsampler = pixelshuffle_block(feature_channels, num_out_ch, upscale_factor=upscale)
        self.cuda()(torch.randn(1, 3, 256, 256).cuda())  # For initialization with GPU tensor

    # Forward pass through the model
    def forward(self, x):
        self.mean = self.mean.type_as(x)  # Ensure mean tensor has same type as input
        x = (x - self.mean) * self.img_range  # Normalize input image
        out_feature = self.conv_1(x)  # Initial convolution

        # Sequentially apply each attention-based block
        out_b1, out_b0_2, att1 = self.block_1(out_feature)
        out_b2, out_b1_2, att2 = self.block_2(out_b1)
        out_b3, out_b2_2, att3 = self.block_3(out_b2)
        out_b4, out_b3_2, att4 = self.block_4(out_b3)
        out_b5, out_b4_2, att5 = self.block_5(out_b4)
        out_b6, out_b5_2, att6 = self.block_6(out_b5)

        out_final = self.conv_2(out_b6)  # Final convolution on features
        # Concatenate multiple outputs and reduce dimensions
        out = self.conv_cat(torch.cat([out_feature, out_final, out_b1, out_b5_2], 1))
        output = self.upsampler(out)  # Upsample to increase resolution
        return output

# Main function to analyze FLOPs and measure runtime
if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table  # Library for FLOP analysis
    import time

    # Initialize model, set to evaluation mode, and prepare random input
    model = SPAN30(3, 3, upscale=4, feature_channels=48).cuda()
    model.eval()
    inputs = (torch.rand(1, 3, 256, 256).cuda(),)  # Create random input for testing
    print(flop_count_table(FlopCountAnalysis(model, inputs)))  # Print FLOP count

    total_time = 0
    input_x = torch.rand(1, 3, 512, 512).cuda()  # Larger input for runtime test
    for i in range(100):  # Run model 100 times for average timing
        torch.cuda.empty_cache()  # Clear CUDA cache for memory
        sta_time = time.time()  # Record start time
        model(input_x)           # Run forward pass
        one_time = time.time() - sta_time  # Calculate duration
        total_time += one_time * 1000      # Convert to milliseconds and add
        print("idx: {} one time: {:.4f} ms".format(i, one_time))  # Print each run time
    print("Avg time: {:.4f}".format(total_time / 100.0))  # Print average time
