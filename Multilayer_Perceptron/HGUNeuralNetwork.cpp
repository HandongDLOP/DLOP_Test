#include "HGULayer.h"
#include "HGUNeuralNetwork.h"

#include <iostream>

int HGUNeuralNetwork::Propagate(float *pInput) {
    if (IsAllocated() == false) {
        std::cout << "HGUNeuralNetwork was not allocated!" << std::endl;
        return false;
    }

    m_aLayer[0].Propagate(pInput);

    for (int i = 1; i < m_noLayer; i++) m_aLayer[i].Propagate(m_aLayer[i - 1].GetOutput());

    for (int j = 0; j < m_aLayer[m_noLayer - 1].GetOutputDim(); j++) std::cout << m_aLayer[m_noLayer - 1].GetOutput()[j] << " ";

    std::cout << '\n';

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

int HGUNeuralNetwork::ComputeGradient(float *pInput, short *pDesiredOutput) {
    if (IsAllocated() == false) {
        std::cout << "HGUNeuralNetwork was not allocated!\n" << std::endl;
        return false;
    }

    Propagate(pInput);

    for (int i = m_noLayer - 1; i >= 0; i--) {
        if (i == m_noLayer - 1) m_aLayer[i].ComputeDeltaBar(pDesiredOutput);
        else m_aLayer[i + 1].Backpropagate(m_aLayer[i].GetDeltaBar());

        m_aLayer[i].ComputeGradientFromDeltaBar();  // problem occured
    }

    return true;
}

int HGULayer::ComputeDeltaBar(short *pDesiredOutput) {
    // if (GetMaxOutputIndex() != std::max_element(pDesiredOutput, pDesiredOutput + m_outputDim) - pDesiredOutput) {
    //
    // } else std::cout << "no learning" << '\n';

    for (int o = 0; o < m_outputDim; o++) {
        m_aDeltaBar[o] = (m_aOutput[o] - pDesiredOutput[o]) / m_outputDim;
    }

    return true;
}

int HGULayer::ComputeGradientFromDeltaBar() {
    int i = 0, o = 0;

    // compute delta from delta_bar
    for (o = 0; o < m_outputDim; o++) m_aDelta[o] = m_aDeltaBar[o] * DerActivationFromOutput(m_aOutput[o]);  // problem occured

    // compute gradient from delta and input
    for (o = 0; o < m_outputDim; o++) {
        for (i = 0; i < m_inputDim; i++) m_aGradient[(m_inputDim + 1) * o + i] += m_aDelta[o] * m_pInput[i];
        m_aGradient[(m_inputDim + 1) * o + m_inputDim] += m_aDelta[o];  // bias
    }

    return true;
}

int HGULayer::Backpropagate(float *pPrevDeltaBar) {
    for (int i = 0; i < m_inputDim; i++) {
        pPrevDeltaBar[i] = 0.F;

        for (int o; o < m_outputDim; o++) pPrevDeltaBar[i] += m_aWeight[(m_inputDim + 1) * o + i] * m_aDelta[o];
    }

    return true;
}

int HGUNeuralNetwork::UpdateWeight(float learningRate) {
    if (IsAllocated() == false) {
        std::cout << "HGUNeuralNetwork was not allocated!\n" << std::endl;
        return false;
    }

    for (int i = 0; i < m_noLayer; i++) {
        FILE *pFile = fopen("Weight.csv", "at");

        fprintf(pFile, "no_Layer: %d,", i);

        fclose(pFile);

        m_aLayer[i].UpdateWeight(learningRate);
    }
    return true;
}

int HGULayer::UpdateWeight(float learningRate) {
    FILE *pFile = fopen("Weight.csv", "at");

    for (int o = 0; o < m_outputDim; o++) {
        fprintf(pFile, "m_outputDim : %d ,", o);

        for (int i = 0; i < m_inputDim + 1; i++) {
            fprintf(pFile, "m_inputDim : %d ,", i);

            fprintf(pFile, "%f,",               m_aWeight[o * (m_inputDim + 1) + i]);

            m_aWeight[o * (m_inputDim + 1) + i]  -= learningRate * m_aGradient[o * (m_inputDim + 1) + i];
            m_aGradient[o * (m_inputDim + 1) + i] = 0.F;  // reset gradient
        }
    }

    fclose(pFile);

    return true;
}
