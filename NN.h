/* NN.h --- NN
 */

#ifndef INCLUDED_NN_H
#define INCLUDED_NN_H 1

#include "AbstractLayer.h"
#include "Layer.h"
#include "ConvolutionLayer.h"
#include "SamplingLayer.h"

class NN {
 public:
  NN();
  NN(int height, int width, int labelcount, int samplecount, int testcount);
  ~NN();

  void add(AbstractLayer *&layer);
  void loadTrainingDataFromCSV(char *s);
  void loadTestDataFromCSV(char *s);

  void train(double alpha, int step);
  void randomInitWeight(void);
  void forward(int k);
  void backpropagate(int k);
  void update(double alpha);
  void applyResult(void);
  void writeToFile(char *s);
  void check(int k);
  int validate(void);
  void validate(double &error);
 private:
  int m_layercount;
  std::vector<AbstractLayer *> m_layers;
  int m_samplecount;
  int m_height;
  int m_width;
  int m_labelcount;
  int m_testcount;
  double ***m_x;
  double **m_y;
  double ***m_xx;
};

#endif /* INCLUDED_NN_H */

