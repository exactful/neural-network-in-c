# Neural network in C

This repository contains a C program for training a neural network using backpropagation on the MNIST dataset. The code is designed to implement forward propagation, backward propagation, weight updates, and performance evaluation through confusion matrix generation.

## About the dataset

The MNIST digits dataset (8x8) is a widely used benchmark dataset in the field of machine learning and computer vision. It consists of grayscale images of handwritten digits from 0 to 9, each digit being represented as an 8x8 pixel image. This dataset has been extensively used for training and testing classification algorithms due to its simplicity and accessibility.

## Requirements

- C compiler
- MNIST dataset via mnist-input.csv and mnist-output.csv

## Code structure

The main components of the code include:

- `main()`: The main training loop that loads the dataset, creates the neural network layers and performs training epochs
- `create_layer()`: Creates layers with neurons and weights
- `load_file_data()`: Reads data from CSV files into arrays
- `do_forward_propagation()`: Computes forward pass through the network
- `compute_node_deltas()`: Computes node deltas using backpropagation
- `update_weights()`: Updates weights using gradient descent
- `display_weights()`: Displays the neural network weights
- `display_predictions()`: Displays input, output, and predicted values
- `display_confusion_matrix()`: Generates and displays the confusion matrix

## Architecture

- Input layer: 64 inputs (based on 8x8 images)
- 2 hidden layers: 32 nodes and 16 nodes
- Output layer: 10 outputs (numbers 0-9 one hot encoded)

## Hyperparameters

- Learning rate: 0.01
- Epochs: 5,000

## Usage

```
gcc main.c -o train.out
./train.out
```

## Example training

```
Epoch      Loss 2.07595801
Epoch  100 Loss 0.12041770
Epoch  200 Loss 0.05075743
Epoch  300 Loss 0.03177120
Epoch  400 Loss 0.02303822
Epoch  500 Loss 0.01801618
Epoch  600 Loss 0.01438107
Epoch  700 Loss 0.01137713
Epoch  800 Loss 0.00920074
Epoch  900 Loss 0.00764972
Epoch 1000 Loss 0.00656634
Epoch 1100 Loss 0.00574113
Epoch 1200 Loss 0.00512238
Epoch 1300 Loss 0.00467642
Epoch 1400 Loss 0.00433394
Epoch 1500 Loss 0.00405346
Epoch 1600 Loss 0.00380757
Epoch 1700 Loss 0.00358352
Epoch 1800 Loss 0.00338427
Epoch 1900 Loss 0.00321358
Epoch 2000 Loss 0.00306663
Epoch 2100 Loss 0.00291770
Epoch 2200 Loss 0.00255454
Epoch 2300 Loss 0.00237196
Epoch 2400 Loss 0.00225338
Epoch 2500 Loss 0.00216143
Epoch 2600 Loss 0.00208544
Epoch 2700 Loss 0.00202038
Epoch 2800 Loss 0.00196309
Epoch 2900 Loss 0.00191092
Epoch 3000 Loss 0.00185650
Epoch 3100 Loss 0.00151272
Epoch 3200 Loss 0.00135592
Epoch 3300 Loss 0.00128786
Epoch 3400 Loss 0.00123990
Epoch 3500 Loss 0.00120144
Epoch 3600 Loss 0.00116875
Epoch 3700 Loss 0.00114006
Epoch 3800 Loss 0.00111439
Epoch 3900 Loss 0.00109111
Epoch 4000 Loss 0.00106981
Epoch 4100 Loss 0.00105015
Epoch 4200 Loss 0.00103184
Epoch 4300 Loss 0.00101469
Epoch 4400 Loss 0.00099835
Epoch 4500 Loss 0.00098214
Epoch 4600 Loss 0.00096229
Epoch 4700 Loss 0.00050486
Epoch 4800 Loss 0.00044272
Epoch 4900 Loss 0.00041815
```

## Example confusion matrix

```
Confusion matrix, actual (vertical) vs predicted (horizontal):
178 0 0 0 0 0 0 0 0 0 
0 182 0 0 0 0 0 0 0 0 
0 0 177 0 0 0 0 0 0 0 
0 0 0 182 0 1 0 0 0 0 
0 0 0 0 181 0 0 0 0 0 
0 0 0 0 0 182 0 0 0 0 
0 0 0 0 0 0 181 0 0 0 
0 0 0 0 0 0 0 179 0 0 
1 0 0 0 0 0 0 0 173 0 
0 0 0 0 0 0 0 0 1 179
```