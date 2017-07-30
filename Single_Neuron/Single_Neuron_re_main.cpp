#include "Single_Neuron_re.hpp"

int main(int argc, char const *argv[]) {

    // Input Data
    double input          = 1.0;
    double desired_output = 4.0;

    // Model Define
    Neuron Single_Neuron;

    // Initialize Weight and Bias
    Single_Neuron.InitializeWeightandBias();

    // Learning Graph
    for (int count = 0; count < 100; count++) {
        // ForwardPropagation
        ActivationFunction().ReLU(Single_Neuron, input);

        // Output Check
        Single_Neuron.NeuronStatus();

        // BackPropagation
        BackPropagation().GradientDescent(Single_Neuron, desired_output, 0.1);
    }

    // 학습 결과 확인
    Single_Neuron.NeuronStatus();

    return 0;
}
