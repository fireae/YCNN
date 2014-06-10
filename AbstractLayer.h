/* AbstractLayer.h --- AbstractLayer
 */

#ifndef INCLUDED_ABSTRACTLAYER_H
#define INCLUDED_ABSTRACTLAYER_H 1

#include "Share.h"

class AbstractLayer {
 public:
  AbstractLayer();
  AbstractLayer(LayerType type, int plaincount, int height, int width, int pheight, int pwidth);
  virtual ~AbstractLayer();

  LayerType type(void);
  int plain(void);
  int height(void);
  int width(void);

  virtual void addConnection(int table[][2]);
  virtual void randomInitWeight(void);
  virtual void forward(AbstractLayer *&alayer);
  virtual void backpropagate(AbstractLayer *&alayer);
  virtual void update(double alpha, AbstractLayer *&layer);
  virtual void print(void);

  double ***m_a;
  double *m_beta;
  double ***m_bias;
  double ***m_delta;
  double ***m_kernel;
  double *m_pbias;
  std::vector<int> *m_table;
  double *****m_theta;

  LayerType m_type;
  int m_plaincount;
  int m_height;
  int m_width;
  int m_pheight;
  int m_pwidth;

  int m_kernelcount; // number of kernel functions in convolution
  int m_kheight;
  int m_kwidth;
  int m_size; // sampling window size
};

#endif /* INCLUDED_ABSTRACTLAYER_H */

