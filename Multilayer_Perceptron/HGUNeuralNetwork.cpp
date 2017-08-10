#include "HGULayer.h"
#include "HGUNeuralNetwork.h"

#include <iostream>

using namespace dlop;

void HGUNeuralNetwork::AllocEachLayer(int LayerNumber, int inputDim,
                                      int outputDim) {
    m_aLayer[LayerNumber].Alloc(inputDim, outputDim);
}

int HGULayer::InitialWeightandBias() {
    // random device class instance, source of 'true' randomness for
    // initializing random seed
    std::random_device rd;

    // Mersenne twister PRNG, initialized with seed from previous random
    // device instance
    std::mt19937 gen(rd());

    for (int o = 0; o < m_outputDim; o++) {
        for (int i = 0; i < m_inputDim; i++) {
            std::normal_distribution<float> rand(0, 0.6);
            m_aWeight[o * (m_inputDim + 1) + i] = rand(gen);
        }
        m_aWeight[o * (m_inputDim + 1) + m_inputDim] = 1;  // bias
    }
    return true;
}

int HGUNeuralNetwork::Propagate(float *pInput) {
    if (IsAllocated() == false) {
        std::cout << "HGUNeuralNetwork was not allocated!" << std::endl;
        return false;
    }

    m_aLayer[0].Propagate(pInput);

    for (int i = 1; i < m_noLayer; i++) m_aLayer[i].Propagate(
            m_aLayer[i - 1].GetOutput());

    return true;
}

int HGULayer::Propagate(float *pInput) {
    m_pInput = pInput;  // for training

    for (int o = 0; o < m_outputDim; o++) {
        float  net      = 0.F;
        float *inWeight = m_aWeight + o * (m_inputDim + 1);

        for (int i = 0; i < m_inputDim; i++) net += pInput[i] * inWeight[i];
        net += inWeight[m_inputDim];

        m_aOutput[o] = Activation(net);
    }  // fully connected node

    return true;
}

float HGULayer::Activation(float net) {
    return 1.F / (1.F + (float)exp(-net));
}

int HGUNeuralNetwork::ComputeGradient(float *pInput, short *pDesiredOutput) {
    if (IsAllocated() == false) {
        std::cout << "HGUNeuralNetwork was not allocated!\n" << std::endl;
        return false;
    }

    Propagate(pInput);

    for (int i = m_noLayer - 1; i >= 0; i--) {
        if (i == m_noLayer - 1) m_aLayer[i].ComputeDeltaBar(pDesiredOutput);
        else m_aLayer[i + 1].Backpropagate(m_aLayer[i].GetDeltaBar());

        m_aLayer[i].ComputeGradientFromDeltaBar();
    }

    return true;
}

int HGULayer::ComputeDeltaBar(short *pDesiredOutput) {
    for (int o = 0; o < m_outputDim; o++) {
        m_aDeltaBar[o] = (m_aOutput[o] - pDesiredOutput[o]) / m_outputDim;
    }

    return true;
}

int HGULayer::Backpropagate(float *pPrevDeltaBar) {
    for (int i = 0; i < m_inputDim; i++) {
        pPrevDeltaBar[i] = 0.F;

        for (int o; o < m_outputDim; o++) {
            pPrevDeltaBar[i] += m_aWeight[(m_inputDim + 1) * o + i] * m_aDelta[o];
        }
    }

    return true;
}

int HGULayer::ComputeGradientFromDeltaBar() {
    int i = 0, o = 0;

    // compute delta from delta_bar
    for (o = 0; o < m_outputDim; o++) {
        m_aDelta[o] = m_aDeltaBar[o] * DerActivationFromOutput(m_aOutput[o]);
    }

    // compute gradient from delta and input
    for (o = 0; o < m_outputDim; o++) {
        for (i = 0; i < m_inputDim; i++) {
            m_aGradient[(m_inputDim + 1) * o + i] += m_aDelta[o] * m_pInput[i];
        }

        m_aGradient[(m_inputDim + 1) * o + m_inputDim] += m_aDelta[o];  // bias
    }

    return true;
}

float HGULayer::DerActivationFromOutput(float output) {
    return output * (1.F - output);
}

int HGUNeuralNetwork::UpdateWeight(float learningRate) {
    if (IsAllocated() == false) {
        std::cout << "HGUNeuralNetwork was not allocated!\n" << std::endl;
        return false;
    }

    for (int i = 0; i < m_noLayer; i++) {
        m_aLayer[i].UpdateWeight(learningRate);
    }
    return true;
}

int HGULayer::UpdateWeight(float learningRate) {
    for (int o = 0; o < m_outputDim; o++) {
        for (int i = 0; i < m_inputDim + 1; i++) {
            m_aWeight[o * (m_inputDim + 1) + i]  -= learningRate * m_aGradient[o * (m_inputDim + 1) + i];
            m_aGradient[o * (m_inputDim + 1) + i] = 0.F;  // reset gradient
        }
    }

    return true;
}

void HGUNeuralNetwork::GetResult(float *pInput) {

    Propagate(pInput);

    for (int j = 0; j < m_aLayer[m_noLayer - 1].GetOutputDim(); j++) {
        std::cout << m_aLayer[m_noLayer - 1].GetOutput()[j] << " ";
    }

    std::cout << '\n';
}
