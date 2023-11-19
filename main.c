#include <stdio.h>
#include <stdlib.h>
#include <time.h>

struct Neuron {
	float z;
    float a;
    int num_weights;
    struct Weight *weights;
};

struct Weight {
    float w;
};

struct Layer {
	int num_neurons;
	struct Neuron *neurons; 
};

struct Layer create_layer(int num_neurons, int num_neurons_next_layer);

int main() {

    int num_layers;

    // Capture the number of layers; 3 layers = 1 input, 1 hidden and 1 output
    printf("Enter the total number of layers including the input and output layers:\n");
    scanf("%d", &num_layers);

    // Allocate memory for an array of integers, ready to store the number of neurons in each layer 
    int* num_neurons = (int*) malloc(num_layers * sizeof(int));
    
    int i, j, k;

    // Capture the number of neurons per layer
    for (i=0; i<num_layers; i++) {
        printf("Enter number of neurons in layer %d:\n", i);
        scanf("%d", &num_neurons[i]);
    }

    // Allocate memory for an array of layers, ready to store each layer
    struct Layer* layers = (struct Layer*) malloc(num_layers * sizeof(struct Layer));

    // Create layers
    for (i=0; i<num_layers; ++i) {

        // Create layer with its neurons
        // Neuron weights connect to neurons in the next layer
        layers[i] = create_layer(num_neurons[i], (i<num_layers-1)?num_neurons[i+1]:0);
        printf("Layer %d: %d neurons\n", i+1, num_neurons[i]);
    }

    // Display network
    for (i=0; i<num_layers; ++i) {

        printf("Layer %d: %d neurons\n", i+1, layers[i].num_neurons);

        for (j=0; j<layers[i].num_neurons; ++j) {

            printf("\tNeuron %d | a: %f z: %f weights: %d\n", j+1, layers[i].neurons[j].a, layers[i].neurons[j].z, layers[i].neurons[j].num_weights);
            for (k=0; k<layers[i].neurons[j].num_weights; ++k) {
                printf("\t\tWeight: %f ", layers[i].neurons[j].weights[k].w);
            }
            printf("\n");
        }
    }

    // TODO: Load data
    // TODO: Forward pass
    // TODO: Training

    // Free memory allocations
    free(num_neurons);
    free(layers);

    // TODO: free other memory allocations
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