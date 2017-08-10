#ifndef _HGUNeuralNetwork_H_
#define _HGUNeuralNetwork_H_

#include "HGULayer.h"

namespace dlop {
/*
 * @Definition   : 전체 NeuralNetwork 구성 요소 및 작동 메커니즘 정의
 *
 * @Structure    : Propagate
 *                 ComputeGradient
 *                 updateWeight
 *
 * @Member       : Number of Layer
 *                 Layer
 *
 */
class HGUNeuralNetwork {
private:

    // @Member
    int m_noLayer;
    HGULayer *m_aLayer;

public:

    HGUNeuralNetwork(int p_noLayer) {
        Alloc(p_noLayer);
    }

    ~HGUNeuralNetwork() {
        Delete();
    }

    int IsAllocated() {
        return m_aLayer != NULL;
    }

    /*
     * @brief Allocation array of Layer (using number of Layer)
     *
     * @param noLayer
     *        레이어 층의 개수를 결정한다.
     */
    int Alloc(int noLayer) {
        m_noLayer = noLayer;
        m_aLayer  = new HGULayer[noLayer];

        return true;
    }

    void Delete() {
        delete[] m_aLayer;
    }

    /*
     * @brief Allocation each Layer (with HGULayer::Alloc)
     *
     * @param LayerNumber
     *        몇 번째 레이어에 관한 Allocation인지를 결정한다.
     * @param inputDim
     *        레이어의 입력 차원을 입력한다.
     * @param outputDim
     *        레이어의 출력 차원을 입력한다.
     */
    void AllocEachLayer(int LayerNumber,
                        int inputDim,
                        int outputDim);

    /*
     * @brief 전체 모델 레벨에서의 ForwardPropagation을 명령한다 (직접 계산 X)
     *
     * @param pInpput
     *        feed로 주어지는 Input값으로 no.o layer의 입력차원과 같은 차원.
     */
    int Propagate(float *pInput);

    /*
     * @brief 전체 모델 레벨에서의 Gradient 계산을 명령한다 (직접 계산 X)
     *
     * @param pInpput
     *        feed로 주어지는 Input값으로 '입력' layer의 입력차원과 같은 차원.
     *        내부의 Propagate를 실행하기 위한 parameter
     * @param pDesiredOutput
     *        학습을 위한 라벨값으로 '출력' Layer의 출력차원과 같은 차원이다.
     */
    int ComputeGradient(float *pInput,
                        short *pDesiredOutput);

    /*
     * @brief 전체 모델 레벨에서의 Weight update를 명령한다 (직접 계산 X)
     *
     * @param learningRate
     *        학습에서 사용되는 LearningRate 값이다.
     */
    int UpdateWeight(float learningRate);
};
}

#endif // _HGUNeuralNetwork_H_
