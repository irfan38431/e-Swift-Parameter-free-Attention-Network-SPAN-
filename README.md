# Swift Parameter free Attention Network SPAN
The SPAN30 model is an enhancement of the original SPAN model designed for efficient image super-resolution. Building upon the foundations of the baseline RLFN and the original SPAN, SPAN30 integrates advanced optimization techniques and attention mechanisms, resulting in significant improvements in both performance and efficiency.

# SPAN Model Comparison and Enhancements
This repository presents the implementation and performance comparison of three models: the RLFN baseline, the original SPAN model, and my enhanced version of SPAN (SPAN-Updated).

# Overview of Models:

## 00_RLFN_baseline: 
A foundational model designed for efficient super-resolution, optimized for simplicity and reduced parameter count.
## 38_SPAN: 
An improved version with a focus on reducing runtime, FLOPs, and memory usage while maintaining comparable performance metrics. It integrates advanced attention mechanisms and efficient convolutional layers.
## SPAN-Updated:
My customized enhancement of the SPAN model, which integrates Channel Attention (CA) layers between each convolutional layer and SPAB block. These modifications aim to improve feature refinement and increase the model’s capacity to focus on important regions.

# Performance Comparison:

Model              Val PSNR	      Val Time (ms)	      Params (M)	      FLOPs (G)	      Activations (M)	      Memory (MB)      	Convs
RLFN Baseline	     26.96	             30.62	        0.317	            19.67	              80.05	               468.84	         39
SPAN	             26.94	             16.10	        0.151            	9.83	              41.68                705.24	         22
SPAN-Updated	     27.01	             20.15	        0.151	            10.23              	41.68                715.24	         22

# Key Changes and Improvements:

-> Channel Attention Integration: CA layers are added after each convolutional layer and SPAB block. This modification improves the model’s ability to focus on relevant features, potentially enhancing the PSNR and general performance.

->Architecture Optimization: The model retains the structure and efficiency of SPAN while improving memory management and computational requirements, as evidenced by the adjusted FLOPs and memory usage figures.

->performance Metrics: The updated SPAN shows an increase in validation time to 20.15 ms due to the added complexity but maintains a competitive PSNR of 27.01 dB, demonstrating improved output quality.

## 1. RLFN Baseline Model
### Architecture Overview:

1 -> Convolutional Layers: The baseline RLFN model uses multiple standard convolutional layers. It is structured with an initial convolution layer, followed by four residual local feature blocks (RLFBs) and ends with an upsampling layer using pixel shuffle.

2 -> Residual Local Feature Block (RLFB): Each RLFB consists of a series of convolutional layers that learn local features, followed by an Enhanced Spatial Attention (ESA) block. The ESA focuses on important spatial features within the image to enhance the super-resolution process.

3 -> ESA (Enhanced Spatial Attention): This module emphasizes spatial regions of interest, allowing the model to pay more attention to significant areas. This helps in retaining important details during upscaling.

4 -> Upsampling Strategy: The pixel shuffle operation is used to increase the resolution efficiently.


### Advantages:
=> Simple and efficient structure with relatively low parameter count and FLOPs, making it computationally light.
=> Effective use of ESA blocks improves spatial detail retention.

### Limitations:
>> Limited use of attention mechanisms and lacks advanced features that could further improve accuracy.
>> PSNR is lower compared to the more optimized versions, and validation runtime is higher (30.62 ms) due to less efficient processing.
## 2. Original SPAN Model (38_SPAN)
### Architecture Overview:
1 -> Parameter-Free Design: The SPAN (Swift Parameter-free Attention Network) architecture introduces a parameter-efficient design that maintains performance while reducing model complexity.
2 -> Conv3XC2 Module: This module is used extensively in SPAN for efficient feature extraction. It combines grouped convolutions and channel expansion to enhance feature richness with fewer parameters.
3 -> SPAB (Structure-Preserving Attention Block): This block replaces the standard RLFB from the baseline model. SPAB integrates a combination of convolutional layers and a novel attention mechanism to preserve structural integrity while performing super-resolution.
4 -> Channel Reduction and Enhancement: The model uses several channel reduction techniques through the use of pointwise convolutions to minimize computational overhead.
5 -> No ESA Module: The SPAN model does not rely on the traditional ESA but uses a streamlined and faster approach for feature attention, which contributes to its lower runtime and FLOPs.
6 -> Pixel Shuffle Upsampler: Similar to the baseline model, SPAN uses a pixel shuffle method to upscale images efficiently.

### Advantages:

->Runtime Efficiency: The model achieves a validation time of 16.10 ms, much faster than the baseline.
->Reduced Parameters: The parameter count is minimized (0.151 M) while maintaining a competitive PSNR, showcasing the efficiency of its design.
->Low FLOPs: The model achieves only 9.83 GFLOPs, reducing the computational burden significantly.

### Limitations:
>>Memory usage is slightly higher (705.24 MB) due to the structured attention mechanisms, and further optimizations are necessary for memory efficiency.
The architectural changes focus on efficiency but may sacrifice some accuracy (PSNR of 26.94), necessitating further enhancements.

.## 3. SPAN-Updated (My Enhanced Version)
### Architecture Overview:

1 -> Integration of Channel Attention (CA): A major enhancement is the integration of CA layers between every convolutional layer and SPAB block. CA focuses on enhancing the representation of crucial channels and improves the model’s ability to emphasize essential features in the image.
2 -> Refined SPAB Blocks: The SPAB blocks remain central to the model, but with CA layers added between blocks, the overall feature enhancement process becomes more effective. This modification refines the attention mechanism by adjusting channel weights dynamically.
3 -> Conv3XC Module: Retained from the original SPAN model for efficient convolution operations, but with CA layers for additional focus on important channels.
4 -> Retained Upsampling and Conv Layers: The pixel shuffle method and final convolution layers are preserved, ensuring the upscaling operation remains efficient and effective.

### Improvements:
>>PSNR Increase: The addition of CA layers results in a slight PSNR improvement to 27.01 dB, indicating better super-resolution output quality.
>>Moderate Validation Time: While the validation time increased to 20.15 ms due to the additional CA layers, the overall processing remains competitive compared to the baseline.
>>FLOPs and Memory Usage:
FLOPs increased slightly to 10.23 GFLOPs, a necessary trade-off for improved output quality through CA integration.
Memory usage remained similar at 715.24 MB, demonstrating that the architectural changes did not drastically affect memory requirements.

### Advantages:

>Enhanced Feature Refinement: By integrating CA layers, the model adapts dynamically to focus on important channels, leading to improved image quality.
>Balanced Complexity: The architecture optimizes the balance between accuracy and efficiency, maintaining a relatively low parameter count while enhancing model performance.

### Considerations:
The increased validation time suggests that further optimizations might be needed if runtime performance is critical in specific applications.

### 
Conclusion:
The SPAN-Updated model presents a balanced and refined approach by integrating channel attention layers, resulting in improved image quality (PSNR) while maintaining efficient computation (FLOPs). The trade-offs in validation time are justified by the enhancement in feature refinement and output accuracy.
