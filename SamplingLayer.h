/* SamplingLayer.h --- SamplingLayer
 */

#ifndef INCLUDED_SAMPLINGLAYER_H
#define INCLUDED_SAMPLINGLAYER_H 1

#include "AbstractLayer.h"

class SamplingLayer: public AbstractLayer {
 public:
  SamplingLayer();
  SamplingLayer(int plaincount, int height, int width, int pheight, int pwidth, int size);
  ~SamplingLayer();

  void randomInitWeight(void);
  void forward(AbstractLayer *&layer);
  void backpropagate(AbstractLayer *&layer);
  void update(double alpha, AbstractLayer *&clayer);
  void print(void);
};

#endif /* INCLUDED_SAMPLINGLAYER_H */

