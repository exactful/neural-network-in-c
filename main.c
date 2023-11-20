#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

struct Neuron {
	float z;                // Sum of weighted inputs to neuron
    float a;                // Post activation function
    float delta;            // Aka node delta
    int num_weights;        // Number of weights connected to neurons in next layer
    struct Weight *weights; // Array of weights
};

struct Weight {
    float w;
};

struct Layer {
	int num_neurons;
	struct Neuron *neurons; 
};

struct Layer create_layer(int num_neurons, int num_neurons_next_layer);
float sigmoid(float x);
float sigmoid_derivative(float x);

float inputs[2][4] = {
                        {0, 1.0, 0, 1.0},
                        {0, 0, 1.0, 1.0}
                    };

float outputs[4] = {0, 1.0, 1.0, 1.0};

int num_layers = 4;
int num_neurons[] = {2, 3, 3, 1};
int training_examples = 4;
float learning_rate = 0.1;
int epoch = 0;
int epochs = 10000;
float loss = 0.0;

int main() {

    int i, j, k, m;
    
    struct Layer layers[num_layers];
    
    // Create layers, each with neurons and respective output weights to next layer
    for (i=0; i<num_layers; ++i)
        layers[i] = create_layer(num_neurons[i], (i<num_layers-1)?num_neurons[i+1]:0);

    float tmp = 0.0;

    for (epoch=0; epoch < epochs; ++epoch) {
        
        loss = 0.0;

        for (m=0; m<training_examples; ++m) {
            
            printf("*** Epoch %d, training example %d ***\n", epoch, m);
        
            // Compute forward pass for this training example
            for (i=0; i<num_layers; ++i) {
                for (j=0; j<layers[i].num_neurons; ++j) {

                    // Process neuron j in layer i
                    
                    if (i == 0) {
                        // Input layer; output a is the "input" to the network
                        layers[i].neurons[j].a = inputs[j][m];
                    } else {
                        // Hidden and output layers
                        tmp = 0.0;
                        for (k=0; k<layers[i-1].num_neurons; ++k)
                            tmp += layers[i-1].neurons[k].a * layers[i-1].neurons[k].weights[j].w; // Sum outputs and weights from previous layer

                        layers[i].neurons[j].z = tmp;
                        layers[i].neurons[j].a = sigmoid(layers[i].neurons[j].z);    
                    }
                }
            }

            // Compute loss for this training example
            loss += 0.5 * pow(layers[num_layers-1].neurons[0].a - outputs[m], 2); // TODO: Hardcoded [0], assumes one output only

            // Compute node deltas dL/dz
            for (i=num_layers-1; i>=0; --i) {
                // Start at last layer
                for (j=0; j<layers[i].num_neurons; ++j) {
                    
                    if (i == num_layers-1) {
                        // Last layer
                        layers[i].neurons[j].delta = sigmoid_derivative(layers[i].neurons[j].z) * (layers[i].neurons[j].a - outputs[m]); // TODO: Tidy up here for multiple outputs
                    } else {
                        // Hidden and input layers
                        // Multiple output weights for this neuron by node deltas in next layer to right
                        tmp = 0.0;

                        for (k=0; k<layers[i].neurons[j].num_weights; ++k) {
                            tmp += layers[i].neurons[j].weights[k].w * layers[i+1].neurons[k].delta;
                        }
                        layers[i].neurons[j].delta = sigmoid_derivative(layers[i].neurons[j].z) * tmp;
                    }
                }
            }
            
            // Compute weight deltas dz/dw and gradient descent to weights using dL/dz.dz/dw
            for (i=0; i<num_layers; ++i) {
                for (j=0; j<layers[i].num_neurons; ++j) {
                    for (k=0; k<layers[i].neurons[j].num_weights; ++k) {
                        layers[i].neurons[j].weights[k].w = layers[i].neurons[j].weights[k].w - (learning_rate * layers[i].neurons[j].a * layers[i+1].neurons[k].delta);
                    }
                }
            }

            // Display network info
            for (i=0; i<num_layers; ++i) {
                printf("Layer %d:\n", i);
                for (j=0; j<layers[i].num_neurons; ++j) {
                    printf("\tNeuron %d | z: %f a: %f delta: %f weights out: %d [ ", j+1, layers[i].neurons[j].z, layers[i].neurons[j].a, layers[i].neurons[j].delta, layers[i].neurons[j].num_weights);
                    for (k=0; k<layers[i].neurons[j].num_weights; ++k) {
                        printf("w%d-%d: %f ", j+1, k+1, layers[i].neurons[j].weights[k].w);
                    }
                    printf("]\n");
                }
            }

            printf("Loss: %f\n", loss);

            // TODO: Check logic + Add functions + Add comments
            // TODO: Check randomisation logic
            // TODO: Add support for multiple outputs: y1, y2 etc... 


        }
    }
}

struct Layer create_layer(int num_neurons, int num_neurons_next_layer) {

    // Seed the random number generator with the current time
    srand((unsigned int)time(NULL));

	struct Layer layer;

	layer.num_neurons = num_neurons;

    // Allocate memory for an array of neurons, ready to store each neuron
	layer.neurons = (struct Neuron *) malloc(num_neurons * sizeof(struct Neuron));

    // Create neurons in this layer
    for (int i=0; i<num_neurons; ++i) {
        
        struct Neuron neuron;
        layer.neurons[i] = neuron;
        layer.neurons[i].a = 0.0;
        layer.neurons[i].z = 0.0;
        layer.neurons[i].delta = 0.0;
        
        layer.neurons[i].num_weights = num_neurons_next_layer;

        // Allocate memory for an array of weights, ready to store each weight 
        layer.neurons[i].weights = (struct Weight *) malloc(num_neurons_next_layer * sizeof(struct Weight));

        // Create weights for each neuron in this layer
        for (int j=0; j<num_neurons_next_layer; ++j) {

            struct Weight weight;
            layer.neurons[i].weights[j] = weight;
            layer.neurons[i].weights[j].w = (float)rand() / RAND_MAX; // Between 0 and 1
        }
    }
    
	return layer;
}


// Activiation function
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// Activation derivative function
float sigmoid_derivative(float x) {
    return (1.0 - sigmoid(x)) * sigmoid(x);
}