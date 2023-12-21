/* Define the training data ---- */

const char INPUT_FILE[] = "logic-gate-input.csv";
const char OUTPUT_FILE[] = "logic-gate-output.csv";

const float INPUT_FILE_MAX_VALUE = 1.0; // Used to normalise data
const int INPUT_FILE_MAX_ROW_LENGTH = 20;

const float OUTPUT_FILE_MAX_VALUE = 1.0; // Used to normalise data
const int OUTPUT_FILE_MAX_ROW_LENGTH = 20;

const char DELIMITER[] = ",";

const int NUM_INPUTS = 2;
const int NUM_OUTPUTS = 2;
const int NUM_TRAINING_EXAMPLES = 4;

float X[NUM_INPUTS][NUM_TRAINING_EXAMPLES]; // Used to store input data
float Y[NUM_OUTPUTS][NUM_TRAINING_EXAMPLES]; // Used to store output data

/* Define neural network architecture ---- */

const int NUM_LAYERS = 3; // Layers including the input and output layers
const int NUM_NEURONS[] = {NUM_INPUTS, 3, NUM_OUTPUTS}; // Neurons per layer

/* Define training hyperparameters ---- */

const float LEARNING_RATE = 1.0;
const int MAX_EPOCHS = 10000;