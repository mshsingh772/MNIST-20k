# MNIST Model

![Model Tests](https://github.com/mshsingh772/MNIST-20k/actions/workflows/model_tests.yml/badge.svg)

A lightweight CNN model implementation for MNIST digit classification.

## Model Architecture

The model uses a CNN architecture with the following components:

### Layer Structure
- Input: 1x28x28 (MNIST images)
- Conv Block 1:
  - Conv2d(1, 8, 3) with padding=1
  - BatchNorm2d
  - Dropout2d(0.1)
  - Conv2d(8, 12, 3) with padding=1
  - BatchNorm2d
  - Dropout2d(0.1)
  - MaxPool2d(2, 2) → 14x14

- Conv Block 2:
  - Conv2d(12, 16, 3) with padding=1
  - BatchNorm2d
  - Dropout2d(0.1)
  - Conv2d(16, 20, 3) with padding=1
  - BatchNorm2d
  - Dropout2d(0.1)
  - MaxPool2d(2, 2) → 7x7

- Final Block:
  - Conv2d(20, 24, 3) → 5x5
  - BatchNorm2d
  - Conv2d(24, 28, 3) → 3x3
  - Conv2d(28, 10, 1) → 3x3
  - AvgPool2d(3) → 1x1
  - Log Softmax

### Key Features
- Uses Batch Normalization for better training stability
- Implements Dropout (0.001) for regularization
- Global Average Pooling instead of Fully Connected layers
- Less than 20k parameters for efficiency

## Test Cases

The model architecture is verified through automated tests:

1. **Parameter Count Test**
   - Ensures model has less than 20,000 parameters
   - Promotes efficient architecture design

2. **Batch Normalization Test**
   - Verifies presence of BatchNorm layers
   - Important for training stability

3. **Dropout Test**
   - Confirms use of Dropout2d layers
   - Essential for preventing overfitting

4. **Architecture Test**
   - Checks for either Fully Connected or Global Average Pooling
   - Current implementation uses Global Average Pooling

## Training Logs
Using device: cuda

Dataset sizes:
Training set: 60,000 images
Test set:     10,000 images
Batch size:   64
Training batches: 938
Test batches:     157


Model Parameters: 16,352

Starting training...
------------------------------------------------------------

Epoch [1/14]
Training: 100%|███████████████████████████████████████████████████████████████████████████| 938/938 [00:17<00:00, 54.57it/s]
Testing : 100%|███████████████████████████████████████████████████████████████████████████| 157/157 [00:02<00:00, 69.37it/s]

Metrics:
  Train Loss: 0.2935    Train Accuracy: 91.54%
   Test Loss: 0.0458     Test Accuracy: 98.54%

🔥 New best model saved! (Accuracy: 98.54%)
------------------------------------------------------------

Epoch [2/14]
Training: 100%|███████████████████████████████████████████████████████████████████████████| 938/938 [00:16<00:00, 55.91it/s]
Testing : 100%|███████████████████████████████████████████████████████████████████████████| 157/157 [00:02<00:00, 73.63it/s]

Metrics:
  Train Loss: 0.0694    Train Accuracy: 97.83%
   Test Loss: 0.0375     Test Accuracy: 98.83%

🔥 New best model saved! (Accuracy: 98.83%)
------------------------------------------------------------

Epoch [3/14]
Training: 100%|███████████████████████████████████████████████████████████████████████████| 938/938 [00:16<00:00, 55.83it/s]
Testing : 100%|███████████████████████████████████████████████████████████████████████████| 157/157 [00:02<00:00, 71.60it/s]

Metrics:
  Train Loss: 0.0519    Train Accuracy: 98.39%
   Test Loss: 0.0236     Test Accuracy: 99.24%

🔥 New best model saved! (Accuracy: 99.24%)
------------------------------------------------------------

Epoch [4/14]
Training: 100%|███████████████████████████████████████████████████████████████████████████| 938/938 [00:16<00:00, 55.73it/s]
Testing : 100%|███████████████████████████████████████████████████████████████████████████| 157/157 [00:02<00:00, 72.69it/s]

Metrics:
  Train Loss: 0.0428    Train Accuracy: 98.69%
   Test Loss: 0.0266     Test Accuracy: 99.18%
------------------------------------------------------------

Epoch [5/14]
Training: 100%|███████████████████████████████████████████████████████████████████████████| 938/938 [00:16<00:00, 55.99it/s]
Testing : 100%|███████████████████████████████████████████████████████████████████████████| 157/157 [00:02<00:00, 73.26it/s]

Metrics:
  Train Loss: 0.0407    Train Accuracy: 98.73%
   Test Loss: 0.0267     Test Accuracy: 99.14%
------------------------------------------------------------

Epoch [6/14]
Training: 100%|███████████████████████████████████████████████████████████████████████████| 938/938 [00:16<00:00, 56.04it/s]
Testing : 100%|███████████████████████████████████████████████████████████████████████████| 157/157 [00:02<00:00, 71.41it/s]

Metrics:
  Train Loss: 0.0341    Train Accuracy: 98.94%
   Test Loss: 0.0261     Test Accuracy: 99.16%
------------------------------------------------------------

Epoch [7/14]
Training: 100%|███████████████████████████████████████████████████████████████████████████| 938/938 [00:16<00:00, 55.49it/s]
Testing : 100%|███████████████████████████████████████████████████████████████████████████| 157/157 [00:02<00:00, 74.01it/s]

Metrics:
  Train Loss: 0.0314    Train Accuracy: 98.99%
   Test Loss: 0.0230     Test Accuracy: 99.30%

🔥 New best model saved! (Accuracy: 99.30%)
------------------------------------------------------------

Epoch [8/14]
Training: 100%|███████████████████████████████████████████████████████████████████████████| 938/938 [00:16<00:00, 55.81it/s]
Testing : 100%|███████████████████████████████████████████████████████████████████████████| 157/157 [00:02<00:00, 71.93it/s]

Metrics:
  Train Loss: 0.0286    Train Accuracy: 99.07%
   Test Loss: 0.0210     Test Accuracy: 99.31%

🔥 New best model saved! (Accuracy: 99.31%)
------------------------------------------------------------

Epoch [9/14]
Training: 100%|███████████████████████████████████████████████████████████████████████████| 938/938 [00:16<00:00, 55.75it/s]
Testing : 100%|███████████████████████████████████████████████████████████████████████████| 157/157 [00:02<00:00, 70.28it/s]

Metrics:
  Train Loss: 0.0268    Train Accuracy: 99.16%
   Test Loss: 0.0216     Test Accuracy: 99.35%

🔥 New best model saved! (Accuracy: 99.35%)
------------------------------------------------------------

Epoch [10/14]
Training: 100%|███████████████████████████████████████████████████████████████████████████| 938/938 [00:17<00:00, 55.02it/s]
Testing : 100%|███████████████████████████████████████████████████████████████████████████| 157/157 [00:02<00:00, 71.14it/s]

Metrics:
  Train Loss: 0.0248    Train Accuracy: 99.14%
   Test Loss: 0.0210     Test Accuracy: 99.28%
------------------------------------------------------------

Epoch [11/14]
Training: 100%|███████████████████████████████████████████████████████████████████████████| 938/938 [00:16<00:00, 55.85it/s]
Testing : 100%|███████████████████████████████████████████████████████████████████████████| 157/157 [00:02<00:00, 72.26it/s]

Metrics:
  Train Loss: 0.0256    Train Accuracy: 99.16%
   Test Loss: 0.0191     Test Accuracy: 99.34%
------------------------------------------------------------

Epoch [12/14]
Training: 100%|███████████████████████████████████████████████████████████████████████████| 938/938 [00:16<00:00, 55.58it/s]
Testing : 100%|███████████████████████████████████████████████████████████████████████████| 157/157 [00:02<00:00, 73.40it/s]

Metrics:
  Train Loss: 0.0224    Train Accuracy: 99.24%
   Test Loss: 0.0196     Test Accuracy: 99.37%

🔥 New best model saved! (Accuracy: 99.37%)
------------------------------------------------------------

Epoch [13/14]
Training: 100%|███████████████████████████████████████████████████████████████████████████| 938/938 [00:16<00:00, 55.43it/s]
Testing : 100%|███████████████████████████████████████████████████████████████████████████| 157/157 [00:02<00:00, 72.71it/s]

Metrics:
  Train Loss: 0.0219    Train Accuracy: 99.32%
   Test Loss: 0.0201     Test Accuracy: 99.42%

🔥 New best model saved! (Accuracy: 99.42%)
------------------------------------------------------------

Epoch [14/14]
Training: 100%|███████████████████████████████████████████████████████████████████████████| 938/938 [00:16<00:00, 55.57it/s]
Testing : 100%|███████████████████████████████████████████████████████████████████████████| 157/157 [00:02<00:00, 73.69it/s]

Metrics:
  Train Loss: 0.0208    Train Accuracy: 99.33%
   Test Loss: 0.0187     Test Accuracy: 99.42%
------------------------------------------------------------

Training completed! Best test accuracy: 99.42%