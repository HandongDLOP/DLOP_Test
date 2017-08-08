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

    my_Neuralnet.AllocEachLayer(0, 2, 3);

    my_Neuralnet.AllocEachLayer(1, 3, 1);

    for (int i = 0; i < 1000000; i++) {
        std::cout << "count: " << i << '\n';

        FILE *pFile = fopen("Weight.csv", "at");

        fprintf(pFile, "\n");

        // std::cout << "input " << input[i % 4][0] << "," << input[i % 4][1] <<
        // " : ";

        my_Neuralnet.ComputeGradient(input[i % 4], DesiredOutput[i % 4]);
        my_Neuralnet.UpdateWeight(0.3);

        if (i % 10 == 9) {
            my_Neuralnet.Propagate(input[0]);
            my_Neuralnet.Propagate(input[1]);
            my_Neuralnet.Propagate(input[2]);
            my_Neuralnet.Propagate(input[3]);
        }

        fprintf(pFile, "\n");

        fclose(pFile);
    }

    std::cout << "Result: " << '\n';

    my_Neuralnet.Propagate(input[0]);
    my_Neuralnet.Propagate(input[1]);
    my_Neuralnet.Propagate(input[2]);
    my_Neuralnet.Propagate(input[3]);

    return 0;
}
