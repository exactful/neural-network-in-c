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
float randn();
float sigmoid(float x);
float sigmoid_derivative(float x);
void do_forward_propagation(Layer layers[], int m);
void compute_node_deltas(Layer layers[], int m);
void update_weights(Layer layers[]);
void display_weights(Layer layers[], int m);
void display_predictions(Layer layers[]);
void display_confusion_matrix(Layer layers[]);

/* Define verbose or not; 1 = display additional info, 0 = quiet */

const int VERBOSE = 0; //

/* Define the training data ---- */

const char INPUT_FILE[] = "mnist-input.csv";
const char OUTPUT_FILE[] = "mnist-output.csv";

const float INPUT_FILE_MAX_VALUE = 16.0; // Used to normalise data
const int INPUT_FILE_MAX_ROW_LENGTH = 10000;

const float OUTPUT_FILE_MAX_VALUE = 1.0; // Used to normalise data
const int OUTPUT_FILE_MAX_ROW_LENGTH = 10000;

const char DELIMITER[] = ",";

const int NUM_INPUTS = 64;
const int NUM_OUTPUTS = 10;
const int NUM_TRAINING_EXAMPLES = 1797;

float X[NUM_INPUTS][NUM_TRAINING_EXAMPLES]; // Used to store input data
float Y[NUM_OUTPUTS][NUM_TRAINING_EXAMPLES]; // Used to store output data

/* Define neural network architecture ---- */

const int NUM_LAYERS = 4; // Layers including the input and output layers
const int NUM_NEURONS[] = {NUM_INPUTS, 32, 16, NUM_OUTPUTS}; // Neurons per layer

/* Define training hyperparameters ---- */

const float LEARNING_RATE = 0.01;
const int MAX_EPOCHS = 5000;

/* Begin main program ---- */

int main() {

    int i, j, k, m;

    // Seed the random number generator with the current time
    srand((unsigned int)time(NULL));

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

            // Append MSE for current training example
            for (i=0; i<layers[NUM_LAYERS-1].num_neurons; ++i)
                loss += pow(layers[NUM_LAYERS-1].neurons[i].a - Y[i][m], 2);

            // Compute node deltas dL/dz
            compute_node_deltas(layers, m);

            // Update weights
            update_weights(layers);
            
            // Display weights
            if (VERBOSE == 1)
                display_weights(layers, m);
        }

        // Calculate average loss for this epoch
        loss = loss / NUM_TRAINING_EXAMPLES;

        // Display training info
        if (epoch % 100 == 0)
            printf("Epoch %4.0d Loss %8.8f\n", epoch, loss);
    }

    //display_predictions(layers);
    display_confusion_matrix(layers);
}

struct Layer create_layer(int num_neurons, int num_neurons_next_layer) {

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
            layer.neurons[i].weights[j].w = randn(); // Initialise weights using random numbers between 0 and 1 from a normal distribution
        }
    }
    
	return layer;
}

/* Load data from file */
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
            arr[row][col++] = atof(token) / max_value;
            token = strtok(NULL, DELIMITER);
        }

        row++;
        col = 0;
    }
}

/* Generate a random number from a normal distribution */
float randn() {
    float u1 = rand() / (float)RAND_MAX;
    float u2 = rand() / (float)RAND_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* Activiation function */
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

/* Activation derivative function */
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
                layers[i].neurons[j].delta = sigmoid_derivative(layers[i].neurons[j].z) * (layers[i].neurons[j].a - Y[j][m]);
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

/* Update weights */
void update_weights(Layer layers[]) {

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

/* Display predictions */
void display_predictions(Layer layers[]) {
    
    int m, i;

    // Predict using input data
    for (m=0; m<NUM_TRAINING_EXAMPLES; ++m) {
            
        // Compute forward pass
        do_forward_propagation(layers, m);

        printf("Example %d: ", m+1);

        // Display input, output and predicted values
        for (i=0; i<NUM_INPUTS; ++i)
            printf("x%d: %f ", i+1, X[i][m]);
    
        for (i=0; i<NUM_OUTPUTS; ++i)
            printf("y%d: %f ", i+1, Y[i][m]);
        
        for (i=0; i<NUM_OUTPUTS; ++i)
            printf("pred%d: %f ", i+1, layers[NUM_LAYERS-1].neurons[i].a);

        printf("\n");
    }
}

/* Display confusion matrix */
void display_confusion_matrix(Layer layers[]) {

    int confusion_matrix[NUM_OUTPUTS][NUM_OUTPUTS] = {0};
    int m, i, j;

    // Predict using input data
    for (m=0; m<NUM_TRAINING_EXAMPLES; ++m) {
            
        // Compute forward pass
        do_forward_propagation(layers, m);

        float max_y_value = 0;
        int max_y_index = 0;

        for (i=0; i<NUM_OUTPUTS; ++i) {
            if (Y[i][m] > max_y_value) {
                max_y_value = Y[i][m];
                max_y_index = i;
            }
        }
        
        float max_pred_value = 0;
        int max_pred_index = 0;

        for (i=0; i<NUM_OUTPUTS; ++i) {
            if (layers[NUM_LAYERS-1].neurons[i].a > max_pred_value) {
                max_pred_value = layers[NUM_LAYERS-1].neurons[i].a;
                max_pred_index = i;
            }
        }
        confusion_matrix[max_y_index][max_pred_index]++;
    }

    // Display confusion matrix
    printf("\nConfusion matrix, actual (vertical) vs predicted (horizontal):\n");
    for (i=0; i<NUM_OUTPUTS; ++i) {
        for (j=0; j<NUM_OUTPUTS; ++j) {
            printf("%d ", confusion_matrix[i][j]);
        }
        printf("\n");
    }
}