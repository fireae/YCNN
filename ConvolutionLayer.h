/* ConvolutionLayer.h --- ConvolutionLayer
 */

#ifndef INCLUDED_CONVOLUTIONLAYER_H
#define INCLUDED_CONVOLUTIONLAYER_H 1

#include "AbstractLayer.h"

class ConvolutionLayer: public AbstractLayer {
 public:
  ConvolutionLayer();
  ConvolutionLayer(int plaincount, int height, int width, int pheight, int pwidth, int kernelcount, int kheight, int kwidth);
  ~ConvolutionLayer();

  int kernelcount(void);
  int kheight(void);
  int kwidth(void);

  void addConnection(int table[][2]);
  void randomInitWeight(void);
  void forward(AbstractLayer *&layer);
  void backpropagate(AbstractLayer *&layer);
  void update(double alpha, AbstractLayer *&layer);
  void print(void);
};

#endif /* INCLUDED_CONVOLUTIONLAYER_H */

