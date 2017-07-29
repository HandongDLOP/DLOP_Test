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
 * @ Definition   : Neuron 구성 요소 정의 (without ActivationFunction)
 *
 * @ Structure    : Single_Neuron,
 *                  Propagation,
 *
 * @ Member       : Weight,
                    Bias,
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

    Neuron(){
        SetAllocation();
    }
    virtual ~Neuron(){
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

    // Structure
    double  forwardPropagation(const double& p_input);

    // Getter
    double* GetOutput() {
        return m_output;
    }

    // Setter
    void SetOutput(double p_output) {
        *m_output = p_output;
    }
};


/*
 * @ Definition  : ActivationFunction class 정의
 *
 * @ class Param : Neuron,
 */
class ActivationFunction {
private:

    /*No data*/

public:

    ActivationFunction();
    virtual ~ActivationFunction();

    inline const double ReLU(Neuron p_Neuron);
    inline const double Identity(Neuron p_Neuron);
};


/*
 * @ Definition   : BackPropagation class 정의
 *
 */
class BackPropagation {
private:

    /*No data*/

public:

    BackPropagation();
    virtual ~BackPropagation();

    void                GradientDescent(Neuron        p_Neuron,
                                        const double& desired_output);
    inline const double getActGradient(const double& output);
};


#endif // SINGLE_NEURON_HPP_
