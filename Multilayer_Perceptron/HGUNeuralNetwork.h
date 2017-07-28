#ifndef _HGUNeuralNetwork_H_
#define _HGUNeuralNetwork_H_

#include "HGULayer.h"

class HGUNeuralNetwork {
    int m_noLayer;

public:

    HGULayer *m_aLayer; // array of HGULayers

    HGUNeuralNetwork() {
        m_noLayer = 0;
        m_aLayer  = NULL;
    }

    ~HGUNeuralNetwork() {
        Delete();
    }

    int IsAllocated() {
        return m_aLayer != NULL;
    }

    int Alloc(int noLayer) {
        m_noLayer = noLayer;
        m_aLayer  = new HGULayer[noLayer];

        return true;
    }

    void Delete() {
        delete[] m_aLayer;
    }

    int Propagate(float *pInput);
    int GetMaxOutputIndex() {
        return m_aLayer[m_noLayer - 1].GetMaxOutputIndex();
    }

    int ComputeGradient(float *pInput,
                        short *pDesiredOutput);
    int UpdateWeight(float learningRate);
};

#endif // _HGUNeuralNetwork_H_
