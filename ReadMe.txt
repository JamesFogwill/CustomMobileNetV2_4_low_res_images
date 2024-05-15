Custom MobileNetV2 for Low-Resolution Image Classification (CIFAR-10)
This repository contains a PyTorch implementation of a modified MobileNetV2 architecture designed for efficient image classification on low-resolution images, specifically targeting the CIFAR-10 dataset.

Key Features
Optimized for CIFAR-10: The model architecture has been adjusted to handle 32x32 pixel images effectively.
Data Augmentation: Extensive data augmentation techniques are applied to improve robustness and generalization to low-resolution images.
Reduced Model Size: Modifications to the original MobileNetV2 architecture aim to reduce model size while maintaining accuracy.
Post-Training Quantization: The model can be quantized to further reduce its size for mobile deployment.
Modifications Compared to Original MobileNetV2
Reduced Initial Filters: The first convolutional layer has fewer output channels.
Adjusted Strides: Strides in early layers are reduced to preserve spatial information.
Fewer Bottleneck Layers: The network is shallower to prevent overfitting on smaller datasets.
Reduced Output Channels: The final convolutional layer has fewer output channels.
How to Use
Clone the repository:

Bash
git clone https://your_github_repository_url.git
Use code with caution.
content_copy
Install Dependencies:

Bash
pip install torch torchvision 
Use code with caution.
content_copy
Run the script:

Bash
python main.py
Use code with caution.
content_copy
Choose an option:

Press 1 to train the model. (This may take some time)
Press 2 to evaluate a pre-trained quantized model.
Files
main.py: The main Python script containing the model definition, training, evaluation, and quantization code.
cifar10_mobilenetv2_model.pth: (Optional) The saved state dictionary of the trained model (if you choose to train it).
cifar10_mobilenetv2_quantized.pth: (Optional) The saved state dictionary of the quantized model.