#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

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

/* Declare function prototypes */

struct Layer create_layer(int num_neurons, int num_neurons_next_layer);
void load_file_data(const char filename[], float max_value, int max_row_length, int rows, int cols, float arr[rows][cols]);
float sigmoid(float x);
float sigmoid_derivative(float x);
void do_forward_propagation(Layer layers[], int m);
void compute_node_deltas(Layer layers[], int m);
void update_weights(Layer layers[], int m);
void display_weights(Layer layers[], int m);

/* Define the training data ---- */

const char INPUT_FILE[] = "logic-gate-input.csv";
const char OUTPUT_FILE[] = "logic-gate-output.csv";

const float INPUT_FILE_MAX_VALUE = 1.0; // Used to normalise data
const int INPUT_FILE_MAX_ROW_LENGTH = 20;

const float OUTPUT_FILE_MAX_VALUE = 1.0; // Used to normalise data
const int OUTPUT_FILE_MAX_ROW_LENGTH = 20;

const char DELIMITER[] = ",";

const int NUM_INPUTS = 2;
const int NUM_OUTPUTS = 1;
const int NUM_TRAINING_EXAMPLES = 4;

float X[NUM_INPUTS][NUM_TRAINING_EXAMPLES]; // Used to store input data
float Y[NUM_OUTPUTS][NUM_TRAINING_EXAMPLES]; // Used to store output data

/* Define verbose or not; 1 = display additional info, 0 = quiet */
const int VERBOSE = 0; //

/* Define neural network architecture ---- */

const int NUM_LAYERS = 3; // Layers including the input and output layers
const int NUM_NEURONS[] = {2, 3, 1}; // Neurons per layer

/* Define training hyperparameters ---- */

const float LEARNING_RATE = 0.1;
const int MAX_EPOCHS = 10000;

/* Begin main program ---- */

int main() {

    int i, j, k, m;

    // Load training data into X and Y arrays
    load_file_data(INPUT_FILE, INPUT_FILE_MAX_VALUE, INPUT_FILE_MAX_ROW_LENGTH, NUM_INPUTS, NUM_TRAINING_EXAMPLES, X);
    load_file_data(OUTPUT_FILE, OUTPUT_FILE_MAX_VALUE, OUTPUT_FILE_MAX_ROW_LENGTH, NUM_OUTPUTS, NUM_TRAINING_EXAMPLES, Y);

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
            
            // Compute forward pass
            do_forward_propagation(layers, m);

            // Compute loss
            loss += 0.5 * pow(layers[NUM_LAYERS-1].neurons[0].a - Y[0][m], 2); // TODO: Hardcoded [0], assumes one output only

            /* Compute node deltas dL/dz */
            compute_node_deltas(layers, m);

            /* Update weights */
            update_weights(layers, m);
            
            /* Display weights */
            display_weights(layers, m);

            // TODO: Check logic + Add comments
            // TODO: Check randomisation logic
            // TODO: Add support for multiple outputs: y1, y2 etc... 

        }

        // Calculate average loss for this epoch
        loss = loss / NUM_TRAINING_EXAMPLES;

        // Display training info
        if (epoch % 100 == 0)
            printf("Epoch %4.0d Loss %8.8f\n", epoch, loss);
    }

    // Predict using input data
    for (m=0; m<NUM_TRAINING_EXAMPLES; ++m) {
            
        // Compute forward pass
        do_forward_propagation(layers, m);

        printf("x1: %f x2: %f y: %f prediction: %f\n", X[0][m], X[1][m], Y[0][m], layers[NUM_LAYERS-1].neurons[0].a);
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

// Load data from file
void load_file_data(const char filename[], float max_value, int max_row_length, int rows, int cols, float arr[rows][cols]) {
    
    FILE *file_ptr = fopen(filename, "r");
    
    char row_buffer[max_row_length];
    char *token;

    int row;
    int col;

    row = col = 0;

    while (fscanf(file_ptr, "%s", row_buffer) == 1) {
        
        token = strtok(row_buffer, DELIMITER);
        
        while (token != NULL) {
            printf("%s %d %d\n", token, row, col);
            arr[row][col++] = atof(token) / max_value;
            token = strtok(NULL, DELIMITER);
        }

        row++;
        col = 0;
    }
}

// Activiation function
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// Activation derivative function
float sigmoid_derivative(float x) {
    return (1.0 - sigmoid(x)) * sigmoid(x);
}

/* Compute forward propagation using training example m */
void do_forward_propagation(Layer layers[], int m) {

    // For layer i
    for (int i=0; i<NUM_LAYERS; ++i) {

        // For neuron j in layer i
        for (int j=0; j<layers[i].num_neurons; ++j) {
            
            if (i == 0) {
                // Input layer; neuron output a is the "input" to the network
                layers[i].neurons[j].a = X[j][m];
            } else {
                // Hidden and output layers
                float weighted_input_to_neuron = 0.0;
                for (int k=0; k<layers[i-1].num_neurons; ++k)
                    // Compute weighted inputs coming into neuron j from neurons in previous layer
                    weighted_input_to_neuron += layers[i-1].neurons[k].a * layers[i-1].neurons[k].weights[j].w; // Sum outputs and weights from previous layer

                layers[i].neurons[j].z = weighted_input_to_neuron;
                layers[i].neurons[j].a = sigmoid(weighted_input_to_neuron);    
            }
        }
    }
}

/* Compute node deltas dL/dz using training example m */
void compute_node_deltas(Layer layers[], int m) {

    // For layer i, going from right to left
    for (int i=NUM_LAYERS-1; i>=0; --i) {

        // For neuron j in layer i
        for (int j=0; j<layers[i].num_neurons; ++j) {
            
            if (i == NUM_LAYERS-1) {
                // Last layer
                layers[i].neurons[j].delta = sigmoid_derivative(layers[i].neurons[j].z) * (layers[i].neurons[j].a - Y[0][m]); // TODO: Tidy up here for multiple outputs
            } else {
                // Hidden and input layers
                // Multiple output weights for this neuron by node deltas in next layer to right
                float tmp = 0.0;
                for (int k=0; k<layers[i].neurons[j].num_weights; ++k) {
                    tmp += layers[i].neurons[j].weights[k].w * layers[i+1].neurons[k].delta;
                }
                layers[i].neurons[j].delta = sigmoid_derivative(layers[i].neurons[j].z) * tmp;
            }
        }
    }
}

void update_weights(Layer layers[], int m) {

    // Compute weight deltas dz/dw and gradient descent to weights using dL/dz.dz/dw
    // For layer i
    for (int i=0; i<NUM_LAYERS; ++i) {
        // For neuron j in layer i
        for (int j=0; j<layers[i].num_neurons; ++j) {
            // For weight k in neuron j in layer i
            for (int k=0; k<layers[i].neurons[j].num_weights; ++k) {
                layers[i].neurons[j].weights[k].w = layers[i].neurons[j].weights[k].w - (LEARNING_RATE * layers[i].neurons[j].a * layers[i+1].neurons[k].delta);
            }
        }
    }
}

/* Display neural network weights for training example m*/
void display_weights(Layer layers[], int m) {

    if (VERBOSE == 1) {
        printf("Training example %d\n", m);
        // For layer i
        for (int i=0; i<NUM_LAYERS; ++i) {
            printf("Layer %d:\n", i);
            // For neuron j in layer i
            for (int j=0; j<layers[i].num_neurons; ++j) {
                printf("\tNeuron %d | z: %f a: %f delta: %f weights out: %d [ ", j+1, layers[i].neurons[j].z, layers[i].neurons[j].a, layers[i].neurons[j].delta, layers[i].neurons[j].num_weights);
                // For weight k in neuron j in layer i
                for (int k=0; k<layers[i].neurons[j].num_weights; ++k) {
                    printf("w%d-%d: %f ", j+1, k+1, layers[i].neurons[j].weights[k].w);
                }
                printf("]\n");
            }
        }
    }
}