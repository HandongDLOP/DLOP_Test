/*
 * =====================================================================================
 *
 *       Filename:  test.cpp
 *       Licensed:  MIT
 *    Description:
 *
 *        Version:  1.0
 *        Created:  2017년 07월 28일
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Park.Chun.Myong
 *   Organization:  HandongDLOP
 *
 * =====================================================================================
 */

#ifndef SINGLE_NEURON_HPP_
#define SINGLE_NEURON_HPP_

#include <iostream>

namespace dlop {
/*
 * @Definition   : Neuron 구성 요소 정의 (without ActivationFunction)
 *
 * @Structure    : Single_Neuron,
 *                 Propagation,
 *
 * @Member       : Weight,
 *                 Bias,
 *                 input   // for ForwardPropagation
 *                 output  // for BackPropagation
 *
 */
class SingleNeuron {
private:

    // @Member
    double *m_Weight;
    double *m_Bias;
    double *m_input;
    double *m_output;

public:

    SingleNeuron() {
        SetAllocation();
    }

    ~SingleNeuron() {
        Delete();
    }

    void SetAllocation() {
        m_Weight = new double;
        m_Bias   = new double;
        m_input  = new double;
        m_output = new double;
    }

    void Delete() {
        delete m_Weight;
        delete m_Bias;
        delete m_input;
        delete m_output;
    }

    /*
     * @brief Initialize Weight and Bias
     */
    void   InitializeWeightandBias();

    /*
     * @brief Calculate Sigma = Weight * Input + bias
     *
     * @param p_input
     *        외부로부터 들어온 input 값을 member에 저장하고 이를 사용한다.
     */
    double MakeSigma(const double& p_input);

    /*
     * @brief As Result of BackPropagation, update Weight and Bias
     *
     * @param LearningRate
     *        Weight과 Bias 변화에 영향을 미치는 정도
     * @param Gradient
     *        계산 된 Gradient 값
     */
    void   UpdateWeightandBias(const double& LearningRate,
                               const double& Gradient);

    // output member 주소 반환
    double* GetOutput() {
        return m_output;
    }

    // ForwardPropagation 결과 확인
    void NeuronStatus() {
        std::cout << "input: " << m_input[0] << " result: " << m_output[0] <<
            '\n';
        std::cout << "(W, b) : (" << m_Weight[0] << ", " << m_Bias[0] << ")" <<
            '\n';
    }
};


/*
 * @Definition : ActivationFunction class 정의
 *
 */
class ActivationFunction {
public:

    ActivationFunction() {}

    ~ActivationFunction() {}

    /*
     * @brief With Sigma, Calculate output on RelU ActivationFunction
     *
     * @param p_Neuron
     *        Neuron에서 계산되는 Sigma 값 및 output 주소 참조
     * @param p_input
     *        외부에서 주어지는 input 값(주로 Sigma를 계산하는 데에 쓰인다)
     */
    void ReLU(SingleNeuron& p_Neuron,
              const double& p_input);

    // double Identity(Neuron p_Neuron, const double& p_input);
};


/*
 * @Definition : BackPropagation class 정의
 *
 */
class BackPropagation {
public:

    BackPropagation() {}

    ~BackPropagation() {}

    /*
     * @brief Optimizer 방식으로, GradientDescent 방식 차용
     *
     * @param p_Neuron
     *        Neuron에서 output 주소와 UpdateWeightandBias함수 참조
     * @param desired_output
     *        외부에서 예상하는 desired output으로 Loss를 계산하기 위해 사용
     * @param LearningRate
     *        UpdateWeightandBias함수에서 사용하기 위한 용도
     */
    void GradientDescent(SingleNeuron& p_Neuron,
                         const double& desired_output,
                         const double& LearningRate);

    /*
     * @brief ActivationFunction 중 ReLU함수의 Gradient 계산에 사용
     *
     * @param output
     *        ForwardPropagation에서 만들어진 output을 계산에 참조
     */
    inline const double getReLUGradient(const double& output);
};
} // namespace dlop

#endif // SINGLE_NEURON_HPP_
