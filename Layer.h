/* Layer.h --- AbstractLayer
 */

#ifndef INCLUDED_LAYER_H
#define INCLUDED_LAYER_H 1

#include "AbstractLayer.h"

class Layer: public AbstractLayer {
 public:
  Layer();
  Layer(LayerType type, int plaincount, int height, int width, int pheight, int pwidth);
  ~Layer();

  void randomInitWeight(void);
  void forward(AbstractLayer *&layer);
  void backpropagate(AbstractLayer *&layer);
  void update(double alpha, AbstractLayer *&layer);
  void print(void);
};

#endif /* INCLUDED_LAYER_H */

