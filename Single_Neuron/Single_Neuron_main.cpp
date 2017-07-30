#include "Single_Neuron.hpp"

//using namespace dlop;

int main(int argc, char const *argv[]) {
    // Input Data
    double input          = 1.0;
    double desired_output = 4.0;

    // Model Define
    dlop::SingleNeuron m_SingleNeuron;

    // Initialize Weight and Bias
    m_SingleNeuron.InitializeWeightandBias();

    // Learning Graph
    for (int count = 0; count < 100; count++) {
        std::cout << "count: " << count << '\n';

        // ForwardPropagation
        dlop::ActivationFunction().ReLU(m_SingleNeuron, input);

        // Output Check
        m_SingleNeuron.NeuronStatus();

        // BackPropagation ; LearningRate = 0.1
        dlop::BackPropagation().GradientDescent(m_SingleNeuron, desired_output, 0.1);
    }

    std::cout << "Result: " << '\n';

    // 학습 결과 확인
    m_SingleNeuron.NeuronStatus();

    return 0;
}
