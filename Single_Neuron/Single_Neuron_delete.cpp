/*
 * =====================================================================================
 *
 *       Filename:  test.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  2017년 07월 19일 16시 39분 08초
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */
#include <iostream>

inline const double ReLU(const double A, const double B) {
    double result = 0.0;

    if (A < B) result = B;

    return (const double)result;
}

class Neuron {
    double W_;              // weight of one input
    double b_;              // bias

    double input_, output_; // saved for back-prop

public:

    Neuron() : W_(2.0), b_(1.0) {
        std::cout << "Neuron()" << '\n';
    }

    double feedforward(const double& _input) {
        input_ = _input;

        const double sigma = W_ * input_ + b_;

        output_ = getActivation(sigma);

        return output_;
    }

    void propBackward(const double& target) {
        const double alpha = 0.1; // learning Rate

        const double grad = (output_ - target) * getActGrad(output_);

        W_ -= alpha * grad * input_; // last input_) camr from d(wx+b)/dw = x
        b_ -= alpha * grad * 1.0;    // last 1.0 came from d(wx_b)/dw
    }

    const double getActivation(const double& x) {
        // for linear or identity activation functions
        // return x;

        // for ReLU activation functions
        return ReLU(0.0, x);
    }

    double getActGrad(const double& x) {
        if (x > 0.0) return 1.0;

        return .0;

        // return 1.0;
    }

    void feedforwardAndPrint(const double& input) {
        std::cout << "input: " << input << " result: " << feedforward(input) <<
            '\n';
    }

    void GetWeightAndBias() {
        std::cout << "(W, b) : (" << W_ << ", " << b_ << ")" << '\n';
    }
};

int main(int argc, char const *argv[]) {
    // initialize my_Neuron
    double input_  = 0.0;
    double target_ = 0.0;

    std::cout << "PLEASE MAKE INPUT : ";
    std::cin >> input_;

    std::cout << "PLEASE MAKE DESIRED OUTPUT : ";
    std::cin >> target_;

    std::cout << "initialize : W = 2.0, b = 1.0" << '\n';

    Neuron my_Neuron;

    for (int r = 0; r < 100; r++) {
        std::cout << "Training " << double(r) << '\n';
        my_Neuron.feedforwardAndPrint(input_);
        my_Neuron.propBackward(target_);
        my_Neuron.GetWeightAndBias();
    }

    std::cout << "*****End******" << '\n';

    return 0;
}
