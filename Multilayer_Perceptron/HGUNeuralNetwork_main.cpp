#include "HGUNeuralNetwork.h"
#include "HGULayer.h"

int main(int argc, char const *argv[]) {
    float input[4][2] = { { 0, 0 },
                          { 0, 1 },
                          { 1, 0 },
                          { 1, 1 } };

    short DesiredOutput[4][2] = { { 0 },
                                  { 1 },
                                  { 1 },
                                  { 0 } };


    dlop::HGUNeuralNetwork my_Neuralnet(2);

    my_Neuralnet.AllocEachLayer(0, 2, 4);

    my_Neuralnet.AllocEachLayer(1, 4, 1);

    for (int i = 0; i < 1000000; i++) {
        std::cout << "count: " << i << '\n';

        my_Neuralnet.ComputeGradient(input[i % 4], DesiredOutput[i % 4]);

        my_Neuralnet.UpdateWeight(0.3);
    }

    std::cout << "Result: " << '\n';

    my_Neuralnet.GetResult(input[0]);
    my_Neuralnet.GetResult(input[1]);
    my_Neuralnet.GetResult(input[2]);
    my_Neuralnet.GetResult(input[3]);

    std::cout << "-----------FINISH-----------" << '\n';

    return 0;
}
