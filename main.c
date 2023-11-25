#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/* Define structures ---- */

typedef struct Neuron {
	float z;                // Sum of weighted inputs to neuron
    float a;                // Post activation function
    float delta;            // Aka node delta
    int num_weights;        // Number of weights connected to neurons in next layer
    struct Weight *weights; // Array of weights
} Neuron;

typedef struct Weight {
    float w;
} Weight;

typedef struct Layer {
	int num_neurons;
	struct Neuron *neurons; 
} Layer;

struct Layer create_layer(int num_neurons, int num_neurons_next_layer);
float sigmoid(float x);
float sigmoid_derivative(float x);

/* Define the training data ---- */

const float X[2][4] =   {
                            {0.0, 1.0, 0.0, 1.0},
                            {0.0, 0.0, 1.0, 1.0}
                        };

const float Y[4] =          {0.0, 1.0, 1.0, 1.0};

const int NUM_TRAINING_EXAMPLES = 4; // Number of training examples in X

/* Define neural network architecture ---- */

const int NUM_LAYERS = 4; // Layers including the input and output layers
const int NUM_NEURONS[] = {2, 3, 3, 1}; // Neurons per layer

/* Define training hyperparameters ---- */

const float LEARNING_RATE = 0.1;
const int MAX_EPOCHS = 10000;

/* Begin main program ---- */

int main() {

    int i, j, k, m;
    
    // Define an array of Layer objects
    Layer layers[NUM_LAYERS];
    
    // Create each layer with its neurons and their respective output weights to next layer
    for (i=0; i<NUM_LAYERS; ++i)
        layers[i] = create_layer(NUM_NEURONS[i], (i<NUM_LAYERS-1)?NUM_NEURONS[i+1]:0);

    float loss = 0.0;
    float tmp = 0.0;

    // Main training loop
    for (int epoch=0; epoch<MAX_EPOCHS; ++epoch) {
        
        loss = 0.0;

        for (m=0; m<NUM_TRAINING_EXAMPLES; ++m) {
            
            printf("*** Epoch %d, training example %d ***\n", epoch, m);
        
            // Compute forward pass for this training example
            for (i=0; i<NUM_LAYERS; ++i) {
                for (j=0; j<layers[i].num_neurons; ++j) {

                    // Process neuron j in layer i
                    
                    if (i == 0) {
                        // Input layer; output a is the "input" to the network
                        layers[i].neurons[j].a = X[j][m];
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
            loss += 0.5 * pow(layers[NUM_LAYERS-1].neurons[0].a - Y[m], 2); // TODO: Hardcoded [0], assumes one output only

            // Compute node deltas dL/dz
            for (i=NUM_LAYERS-1; i>=0; --i) {
                // Start at last layer
                for (j=0; j<layers[i].num_neurons; ++j) {
                    
                    if (i == NUM_LAYERS-1) {
                        // Last layer
                        layers[i].neurons[j].delta = sigmoid_derivative(layers[i].neurons[j].z) * (layers[i].neurons[j].a - Y[m]); // TODO: Tidy up here for multiple outputs
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
            for (i=0; i<NUM_LAYERS; ++i) {
                for (j=0; j<layers[i].num_neurons; ++j) {
                    for (k=0; k<layers[i].neurons[j].num_weights; ++k) {
                        layers[i].neurons[j].weights[k].w = layers[i].neurons[j].weights[k].w - (LEARNING_RATE * layers[i].neurons[j].a * layers[i+1].neurons[k].delta);
                    }
                }
            }

            // Display network info
            for (i=0; i<NUM_LAYERS; ++i) {
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
        layer.neurons[i].weights = (Weight *) malloc(num_neurons_next_layer * sizeof(Weight));

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