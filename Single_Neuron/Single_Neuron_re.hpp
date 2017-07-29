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

/*
 * @ Definition   : Model (Graph) 구성 요소 정의
 *
 * @ Structure    : Single_Neuron,
 *                  Propagation,
 *                  Output_from_Graph,
 *
 * @ Member       : Weight,
                    Bias
 *
 */

class Neuron {
private:

    // @ Member
    double *m_Weight;
    double *m_Bias;
    double *m_input;
    double *m_output;

public:

    Neuron();
    virtual ~Neuron();

    // Structure
    const double forwardPropagation(const double& p_input);
};


/*
 * Definition : ActivationFunction class 정의
 *
 * Param      : input, output
 */
class ActivationFunction {
private:

    Neuron *m_Neuron;

public:

    ActivationFunction();
    virtual ~ActivationFunction();

    inline const double ReLU(const double input,
                             const double output);
    inline const double Identity(const double input,
                                 const double output);
};


/*
 * @ Definition   : ActivationFunction class 정의
 *
 * @ Param        : desired_output, output
 *
 * @ Class Member : Model
 */
class BackPropagation {
private:

    Neuron *m_Neuron;

public:

    BackPropagation();
    virtual ~BackPropagation();

    void                GradientDescent(const double& desired_output);
    inline const double getActGradient(const double& output);
};


#endif // SINGLE_NEURON_HPP_
