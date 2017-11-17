#include <stdio.h>


// this is connection operator in main()
Operator Convolution = new Convolution(x, W_conv1, 1, 1, 1, 1); // default padding = 0

class Convolution : public Operator{

public:
// this is Convolution operation in sess.run()
// this function uses the member variables in Operator class.
// input.shape contains [total # of data, row, column, depth]
// weight.shape contains [width, height, channel, # of filters]
bool ForwardPropagate(){
  int maskWidth = m_aWeight.shape(0);
  int maskHeight = m_aWeight.shape(1);
  float net = 0;

  for (int outputPlane = 0; outputPlane < m_aWeight.shape(3); outputPlane++){ // p

    for (int i = 0; i * m_Stride2 < m_aInput.shape(1); i++){
      for (int j = 0; j * m_Stride3 < m_aInput.shape(2); j++){
        for (int inputPlane = 0; inputPlane < m_aWeight.shape(2); inputPlane++){ // q
          float *pMask = m_aWeight.getData(inputPlane, outputPlane);
          net += Convolution(m_aInput.getData(inputPlane), pMask, i, j, maskWidth, maskHeight, m_Stride2, m_Stride3);
        }
        m_aOutput[outputPlane].setData(i, j, activation(net + m_aBias[outputPlane].getData()));
        net = 0;
      }
    }
  }

// refactoring
  for (int outputPlane = 0; outputPlane < m_aWeight.shape(3); outputPlane++){ // p
    for (int inputPlane = 0; inputPlane < m_aWeight.shape(2); inputPlane++){ // q
      float *pMask = m_aWeight.getData(inputPlane, outputPlane);

      for (int i = 0; i < m_aInput.shape(1); i++){
        for (int j = 0; j < m_aInput.shape(2); j++){
          net = Convolution(m_aInput.getData(inputPlane), pMask, i, j, maskWidth, maskHeight, m_Stride2, m_Stride3);
          m_aOutput[outputPlane].addData(i, j, net + m_aBias[outputPlane].getData()));
        }
      }
    }
    elementWiseActivation(m_aOutput[outputPlane]);
  }

}


// i * j is input size,  u * v is mask size
float Convolution(float *x, float *w, int i, int j, int mw, int mh, int sx, int sy){
  float output = 0;

  for (int u = 0; u < mh; u++){
    for (int v = 0; v < mw; v++){
      output += x[(j * sx + v) + ((i * sy + u) * m_aInput.shape(2))] * w[v + (u * mw)];
    }
  }

  return output;
}






// requirements : upper-GradientsDescent
// BP : deliver dE/dInput to bottom layer.
// compute the dE/dInput
Operator Backpropagate(){
  ComputeGradientDescent();
  m_aGradient.setData(m_aDelta * m_aInput.getData());
}

// ComputeGradientDescent(upper G.D.) : UpdateWeight()
void ComputeGradientDescent(){
  Operator **outputOperator = GetOutputOperator();
  Tensor outputDelta = outputOperator.getDelta();
  Tensor outputWeight = outputOperator.getWeight();

  // dE/dX^(n-1)
  float deltaBar = outputDelta.getData() * outputWeight.getData();
  m_aDelta.setData(deltaBar * derivativeActivationFunction(net));
}



//
//
