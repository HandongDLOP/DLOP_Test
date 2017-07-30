#include "Single_Neuron_re.hpp"

int main(int argc, char const *argv[]) {
    double input          = 1.0;
    double desired_output = 4.0;

    Neuron Single_Neuron;

    Single_Neuron.InitializeWeightandBias();

    Single_Neuron.StatusNeuron();


    for (int count = 0; count < 100; count++) {
        ActivationFunction().ReLU(Single_Neuron, input);

        Single_Neuron.StatusNeuron();

        BackPropagation().GradientDescent(Single_Neuron, desired_output, 0.1);
    }

    return 0;
}
