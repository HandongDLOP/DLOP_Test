#include "HGUNeuralNetwork.h"
#include "HGULayer.h"

int main(int argc, char const *argv[]) {
    // 학습 데이터 준비
    float input[4][2] = { { 0, 0 },
                          { 0, 1 },
                          { 1, 0 },
                          { 1, 1 } };

    short DesiredOutput[4][2] = { { 0 },
                                  { 1 },
                                  { 1 },
                                  { 0 } };

    // 모델 정의 (Layer 개수)
    dlop::HGUNeuralNetwork my_Neuralnet(2);

    // 각 Layer 정의 (node 개수)
    my_Neuralnet.AllocEachLayer(0, 2, 4);

    my_Neuralnet.AllocEachLayer(1, 4, 1);

    // 학습 전 상태 확인
    std::cout << "Initial: " << '\n';

    my_Neuralnet.GetResult(input[0]);
    my_Neuralnet.GetResult(input[1]);
    my_Neuralnet.GetResult(input[2]);
    my_Neuralnet.GetResult(input[3]);

    // 학습 정의
    for (int i = 0; i < 1000000; i++) {
        my_Neuralnet.ComputeGradient(input[i % 4], DesiredOutput[i % 4]);

        if (i % 100 == 99) {
            // std::cout << "count: " << i << '\n';
            my_Neuralnet.UpdateWeight(0.3);
        }
    }

    // 학습 후 결과 확인
    std::cout << "Result: " << '\n';

    my_Neuralnet.GetResult(input[0]);
    my_Neuralnet.GetResult(input[1]);
    my_Neuralnet.GetResult(input[2]);
    my_Neuralnet.GetResult(input[3]);

    std::cout << "-----------FINISH-----------" << '\n';

    return 0;
}
