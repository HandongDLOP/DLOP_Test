#include "tensor.h"

class Convolution : public Operator{
  tensor m_aWeight;
  tensor m_aInput;
  tensor m_Stride1;
  tensor m_Stride2;
  tensor m_Stride3;
  tensor m_Stride4;
  tensor m_aOutput;

public:
  bool ForwardPropagate();
  float Convolution(float *x, float *w, int i, int j, int maskWidth, int maskHeight, int strideX, int strideY);
}
