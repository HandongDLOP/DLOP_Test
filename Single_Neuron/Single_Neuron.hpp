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
 * @ Definition   : Neuron 구성 요소 정의 (without ActivationFunction)
 *
 * @ Structure    : Single_Neuron,
 *                  Propagation,
 *
 * @ Member       : Weight,
 *                  Bias,
 *                  input   // for ForwardPropagation
 *                  output  // for BackPropagation
 *
 */

class SingleNeuron {
private:

    // @ Member
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

    void   InitializeWeightandBias();
    double MakeSigma(const double& p_input);            // Sigma = Weight *
                                                        // input + bias
    void   UpdateWeightandBias(const double& LearningRate,
                               const double& Gradient); // Learning Graph

    // Getter
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
 * @ Definition  : ActivationFunction class 정의
 *
 */
class ActivationFunction {
public:

    ActivationFunction() {}

    ~ActivationFunction() {}

    void ReLU(SingleNeuron& p_Neuron,
              const double& p_input);

    // double Identity(Neuron p_Neuron, const double& p_input);
};


/*
 * @ Definition   : BackPropagation class 정의
 *
 */
class BackPropagation {
public:

    BackPropagation() {}

    ~BackPropagation() {}

    void                GradientDescent(SingleNeuron& p_Neuron,
                                        const double& desired_output,
                                        const double& LearningRate);
    inline const double getReLUGradient(const double& output);
};
} // namespace dlop

#endif // SINGLE_NEURON_HPP_
