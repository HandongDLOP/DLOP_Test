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

#include "Single_Neuron_re.hpp"

// For FowardPropagation
void Neuron::InitializeWeightandBias() {
    m_Weight[0] = 2.0;
    m_Bias[0]   = 1.0;
}

// For ForwardPropagation
double ActivationFunction::ReLU(Neuron      & p_Neuron,
                                const double& p_input) {
    double *output = p_Neuron.GetOutput();

    double Sigma = p_Neuron.MakeSigma(p_input);

    if (0.0 < Sigma) {
        *output = Sigma;
        return *output;
    }

    *output = 0.0;
    return *output;
}

// For FowardPropagation
double Neuron::MakeSigma(const double& p_input) {
    *m_input = p_input;

    double Sigma = m_Weight[0] * m_input[0] + m_Bias[0];

    return Sigma;
}

// For BackPropagation
void BackPropagation::GradientDescent(Neuron      & p_Neuron,
                                      const double& desired_output,
                                      const double& LearningRate) {
    double *output        = p_Neuron.GetOutput();
    const double Gradient = (*output - desired_output) * getReLUGradient(*output);

    p_Neuron.UpdateWeightandBias(LearningRate, Gradient);
}

// For BackPropagation
inline const double BackPropagation::getReLUGradient(const double& output) {
    if (output > 0.0) return 1.0;

    return .0;
}

// For BackPropagation
void Neuron::UpdateWeightandBias(const double& LearningRate,
                                 const double& Gradient) {
    // last input_) came from d(wx+b)/dw = x
    m_Weight[0] -= LearningRate * Gradient * m_input[0];

    // last 1.0 came from d(wx_b)/dw
    m_Bias[0] -= LearningRate * Gradient * 1.0;
}
