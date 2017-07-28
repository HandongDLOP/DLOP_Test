#ifndef _HGULayer_HPP_
#define _HGULayer_HPP_

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include <random>
#include <algorithm>

class HGULayer {
    int m_inputDim;
    int m_outputDim;

    float *m_pInput;  // size: m_inputDim
    float *m_aOutput; // size: m_outputDim
    float *m_aWeight; // size: (m_inputDim + 1) * m_outputDim

    // only for training
    float *m_aGradient;
    float *m_aDelta;
    float *m_aDeltaBar;

    // member function

public:

    HGULayer() {}

    ~HGULayer() {
        // Delete();
    }

    int IsAllocated() {
        return m_aWeight != NULL;
    }

    int Alloc(int inputDim, int outputDim) {
        m_inputDim  = inputDim;
        m_outputDim = outputDim;
        m_pInput    = new float[m_inputDim];
        m_aOutput   = new float[m_outputDim];
        m_aWeight   = new float[(m_inputDim + 1) * m_outputDim];
        InitialWeight();
        m_aGradient = new float[m_outputDim];
        m_aDelta    = new float[m_outputDim];
        m_aDeltaBar = new float[m_outputDim];

        return true;
    }

    int InitialWeight() {
        // random device class instance, source of 'true' randomness for
        // initializing random seed
        std::random_device rd;

        // Mersenne twister PRNG, initialized with seed from previous random
        // device instance
        std::mt19937 gen(rd());

        for (int o = 0; o < m_outputDim; o++) {
            for (int i = 0; i < m_inputDim + 1; i++) {
                std::normal_distribution<float> rand(0, 0.6);
                m_aWeight[o * (m_inputDim + 1) + i] = rand(gen);
            }
        }
        return true;
    }

    void Delete() {
        delete[] m_pInput;
        delete[] m_pInput;
        delete[] m_aOutput;
        delete[] m_aWeight;
        delete[] m_aGradient;
        delete[] m_aDelta;
        delete[] m_aDeltaBar;
    }

    int GetOutputDim() {
        return m_outputDim;
    }

    float* GetOutput() {
        return m_aOutput;
    }

    float* GetDeltaBar() {
        return m_aDeltaBar;
    }

    int Propagate(float *pInput);
    int GetMaxOutputIndex() {
        return std::max_element(m_aOutput, m_aOutput + m_outputDim) - m_aOutput;
    }

    int         ComputeDeltaBar(short *pDesiredOutput);
    int         ComputeGradientFromDeltaBar();
    int         Backpropagate(float *pPrevDeltaBar);

    virtual int UpdateWeight(float learningRate);

    float       Activation(float net) {
        return 1.F / (1.F + (float)exp(-net));
    }

    float DerActivationFromOutput(float output) {
        return output * (1.F - output);
    }
};

#endif // _HGULayer_HPP_
