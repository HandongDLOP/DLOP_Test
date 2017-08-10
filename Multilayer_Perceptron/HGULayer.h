#ifndef _HGULayer_HPP_
#define _HGULayer_HPP_

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include <random>
#include <algorithm>

namespace dlop {
/*
 * @Definition   : 레이어 구성 요소 및 작동 메커니즘 정의
 *
 * @Structure    : InitialWeight
 *
 *
 * @Member       : Input Dimension
 *                 Ouput Dimension
 *                 Input saving place
 *                 Output saving place
 *                 Weight saving place
 *                 Gradient saving place
 *                 Delta() saving place
 *                 Deltabar() saving place
 *
 */
class HGULayer {
private:
    // @Member
    int m_inputDim;
    int m_outputDim;

    float *m_pInput;  // size: m_inputDim
    float *m_aOutput;  // size: m_outputDim
    float *m_aWeight;  // size: (m_inputDim + 1) * m_outputDim

    // only for training
    float *m_aGradient;
    float *m_aDelta;
    float *m_aDeltaBar;

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
        m_aGradient = new float[m_outputDim];
        m_aDelta    = new float[m_outputDim];
        m_aDeltaBar = new float[m_outputDim];

        InitialWeightandBias();

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

    // 이 클래스 및 외부에서 사용하게 될 member 변수의 Getter 정의
    int GetOutputDim() {
        return m_outputDim;
    }

    float* GetOutput() {
        return m_aOutput;
    }

    float* GetDeltaBar() {
        return m_aDeltaBar;
    }

    /*
     * @brief memory Allocation이 일어날 때 Weigt과 Bias를 초기화한다.
     *        Weight의 경우는 normal distribution을 따라 정해주며,
     *        Bias의 경우는 작은 상수 값인 1로 지정한다.
     *
     */
    int   InitialWeightandBias();

    /*
     * @brief Layer 레벨에서의 ForwardPropagation을 정의한다 (직접 계산 O)
     *
     * @param pInpput
     *        feed로 주어지는 Input값으로 각 layer의 입력차원과 같은 차원.
     */
    int   Propagate(float *pInput);

    /*
     * @brief ForwardPropagation을 진행할 때 사용하는 Activation Function이
     *        정의 되어 있다.
     *        여기에서는 Sigmoid 함수를 사용한다.
     *
     * @param net = Weight * Input + Bias이다.
     */
    float Activation(float net);

    /*
     * @brief Gradient를 계산하기 위한 Error function을 정의한 것에서 Deltabar를 의미한다.
     *        Deltabar = (output - Desired output) / output Dimension
     *        Output Layer에서만 계산 된다.
     *
     * @param pDesiredOutput
     *        Backpropagate 수행에 필요한 Error정의를 위해 필요한 라벨
     *        (with Supervised learning)
     */
    int   ComputeDeltaBar(short *pDesiredOutput);

    /*
     * @brief 바로 아래의 Layer의 Delta를 현 Layer의 Weight과 Delta를 이용해서
     *        계산한다.
     *
     * @param 바로 아래 Layer의 Delta값이 저장되는 곳의 주소
     */
    int   Backpropagate(float *pPrevDeltaBar);

    /*
     * @brief 우선 Deltabar와 Output값에 대한 활성함수의 미분값을 이용해서 Delta를 계산한다.
     *        그 다음 구해진 Delta값과 input값을 이용해서 Gradient값을 저장한다.
     *        이 때, Bias의 경우는 Delta 값만 저장한다
     *
     */
    int   ComputeGradientFromDeltaBar();

    /*
     * @brief Output값에 대한 활성함수의 미분 값을 계산한다.
     *        여기에서는 Sigmoid의 정의를 따른 식이 정의되어 있다.
     *
     * @param 현 레이어에서 ForwardPropagation의 결과로 나온 Output값
     */
    float DerActivationFromOutput(float output);

    /*
     * @brief 한 번에 Weight(and Bias)을 Update하지 않고 일정한 주기마다 Update할 수 있도록 한다.
     *
     * @param 하이퍼 파라미터인 학습 속도
     */
    int UpdateWeight(float learningRate);
};
}
#endif  // _HGULayer_HPP_
