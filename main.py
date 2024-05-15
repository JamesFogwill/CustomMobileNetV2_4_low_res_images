import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.mobilenetv2 import InvertedResidual
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split

import os
import time


# ---------- Test Function ---------------
# tests a pre trained model on my CustomMobileNetV2 architecture on the CIFAR-10 test set

def testmodel():
    model = torch.load("./Models/Quantised_Models/E150_A90/my_custom_model.pth")
    model = model.to(device)
    model_size = os.path.getsize("./Models/Quantised_Models/E150_A90/my_custom_model.pth") / (1024 * 1024)
    testdata = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    testloader = DataLoader(testdata, batch_size=64, shuffle=False, num_workers=2)
    criterion = nn.CrossEntropyLoss()
    # device = torch.device("cuda")
    start_time = time.time()
    accuracy, accuracy_loss = evaluate_model(model, testloader, device, criterion)
    end_time = time.time()
    print("Accuracy: ", accuracy, "% ", "Accuracy Loss: ", accuracy_loss)
    print("Model size(on disk): ", model_size, "MB")
    print("Evaluation time: ", end_time - start_time, "seconds")


# ---------- DATA AUGMENTATION -----------
def set_transforms():
    batch_size = 64

    # Compose all transforms for training data
    transform = transforms.Compose([
        # Basic Transforms
        transforms.RandomCrop(32, padding=4),  # Add random cropping with padding to preserve information
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),  # Slight rotations to account for orientation variations

        # Color Transforms
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # Adjust brightness, contrast, etc.
        transforms.RandomGrayscale(p=0.2),  # Occasionally convert to grayscale

        # Advanced Transforms (consider experimenting with these)
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Introduce perspective distortions
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Random translations and scaling

        # Transforms to enhance low-resolution features
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.5),

        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # download a training set
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)

    # splitting the training, 90% training, 10% validation
    train_size = int(0.9 * len(train_set))  # 90% for training
    val_size = len(train_set) - train_size  # 10% for validation
    train_subset, val_subset = random_split(train_set, [train_size, val_size])

    # attach training data to the training loader
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    # attach validation data to the validation loader
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Compose transforms for test data
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]), download=True)

    # attach test data to test loader
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return val_loader, test_loader, train_loader, transform


# --------------- Creates CustomMobileNet Class -----------------
class CustomMobileNetV2(nn.Module):
    def __init__(self, num_classes=10):  # CIFAR-10 has 10 classes
        super(CustomMobileNetV2, self).__init__()

        # Layer 1: Initial Convolution
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU6(inplace=True)

        # Bottleneck Layers (2-18)
        self.bottlenecks = nn.Sequential(
            self._make_bottleneck(32, 16, 1, 1),  # Layer 2
            self._make_bottleneck(16, 24, 6, 1),  # Layer 3
            self._make_bottleneck(24, 24, 6, 1),  # Layer 4
            self._make_bottleneck(24, 32, 6, 2),  # Layer 5
            self._make_bottleneck(32, 32, 6, 1),  # Layer 6
            self._make_bottleneck(32, 64, 6, 2),  # Layer 7
            self._make_bottleneck(64, 64, 6, 1),  # Layer 8
            self._make_bottleneck(64, 96, 6, 2),  # Layer 9
            self._make_bottleneck(96, 96, 6, 1),  # Layer 10
            self._make_bottleneck(96, 160, 6, 2),  # Layer 11
        )

        # Layer 12: Final Convolution
        self.conv19 = nn.Conv2d(160, 960, kernel_size=1, stride=1, bias=False)
        self.bn19 = nn.BatchNorm2d(960)

        # Layer 13: Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Layer 14: Final Classification Layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(960, num_classes),
        )

    def _make_bottleneck(self, in_channels, out_channels, expansion, stride):
        return InvertedResidual(in_channels, out_channels, stride, expand_ratio=expansion)

    # forward pass function is the general loop of the network
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bottlenecks(x)
        x = self.relu(self.bn19(self.conv19(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ------------------ Training Function -----------------
def train_model_2(model, criterion, optimizer, trainloader, device, val_loader, scheduler):
    model.train()
    for epoch in range(1):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate accuracy
        accuracy = 100 * correct / total
        validation_accuracy, validation_loss = evaluate_model(model, val_loader, device, criterion)

        # Update the learning rate after each epoch
        scheduler.step(validation_loss)

        print(f"Epoch: {epoch + 1}, Loss: {running_loss / len(trainloader)}, Accuracy: {accuracy}%")


# ------------ Evaluate Function ---------------

# Used to fetch the accuracy of any model then any data loader
# Also has joint use of returning loss which can be used for validation loss
# with the schedular

def evaluate_model(model, loader, device, criterion):
    model.eval()
    total = 0
    correct = 0
    validation_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print("Accuracy: ", accuracy, "%")
    return accuracy, validation_loss


# ----------------- main loop ----------------
def main():
    # gets transforms and all data loaders
    val_loader, test_loader, train_loader, transform = set_transforms()

    model = CustomMobileNetV2()
    model = model.to(device)

    # sets the optimiser, scheduler and criterion and init parameters
    optimiser = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', patience=3, factor=0.85, verbose=True)
    criterion = nn.CrossEntropyLoss()

    # train and evaluate the model
    train_model_2(model, criterion, optimiser, train_loader, device, val_loader, scheduler)
    evaluate_model(model, test_loader, device, criterion)

    # Save the original model dictionary and model
    model_save_path = './Models/Un-Quantised_Models/custom_state_dict.pth'
    torch.save(model.state_dict(), model_save_path)
    torch.save(model, './Models/Un-Quantised_Models/custom_model.pth')

    # Quantization
    model.eval()  # Set the model to evaluation mode before quantization
    model_int8 = torch.quantization.quantize_dynamic(
        model,  # the original model
        {torch.nn.Conv2d, torch.nn.Linear},  # a set of layers to dynamically quantize
        dtype=torch.qint8)  # the target dtype for quantized weights

    # Save the quantized model and dictionary
    model_save_path = './Models/Quantised_Models/quantised_custom_dict.pth'
    torch.save(model_int8.state_dict(), model_save_path)  # Save model_int8, not model
    torch.save(model_int8, './Models/Quantised_Models/quantised_custom_model.pth')
    print("Quantized model saved to the current directory.")

    # Load and evaluate quantized model
    quantized_model = torch.load('./Models/Quantised_Models/quantised_custom_model.pth')
    quantized_model.eval()  # Set to evaluation mode

    print("\nEvaluating the quantized model:")
    evaluate_model(quantized_model, test_loader, device, criterion)


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Set device to GPU
        print("CUDA is available! Training on GPU...")
    else:
        device = torch.device("cpu")  # Set device to CPU
        print("CUDA is not available. Training on CPU...")
    inp = input("Enter 1 to train the model and 2 to test the model: ")
    if inp == '1':
        main()
    else:
        testmodel()
