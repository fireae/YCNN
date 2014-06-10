#include "NN.h"
#include "AbstractLayer.h"
#include "Layer.h"
#include "ConvolutionLayer.h"
#include "SamplingLayer.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  int height = atoi(argv[1]);
  int width = atoi(argv[2]);
  int labelcount = atoi(argv[3]);
  int samplecount = atoi(argv[4]);
  int testcount = atoi(argv[5]);

  NN nn(height, width, labelcount, samplecount, testcount);
  nn.loadTrainingDataFromCSV(argv[6]);
  // nn.loadTestDataFromCSV(argv[7]);

  int table[6][2];

  AbstractLayer *input = new Layer(N, 1, height, width, height, width);
  AbstractLayer *c1 = new ConvolutionLayer(6, 28, 28, height, width, 6, 5, 5);
  for (int i = 0; i < 6; ++i) {
    table[i][0] = i;
    table[i][1] = 0;
  }
  c1->addConnection(table);
  AbstractLayer *s1 = new SamplingLayer(6, 14, 14, 28, 28, 2);
  AbstractLayer *c2 = new ConvolutionLayer(6, 10, 10, 14, 14, 6, 5, 5);
  for (int i = 0; i < 6; ++i) {
    table[i][0] = i;
    table[i][1] = i;
  }
  c2->addConnection(table);
  AbstractLayer *s2 = new SamplingLayer(6, 5, 5, 10, 10, 2);
  /*
  AbstractLayer *c3 = new ConvolutionLayer(6, 1, 1, 5, 5, 6, 5, 5);
  int c3_table[6][2];
  for (int i = 0; i < 6; ++i) {
    c3_table[i][0] = i;
    c3_table[i][1] = i;
  }
  c3->addConnection(c3_table);
  */
  AbstractLayer *hidden = new Layer(N, 1, 200, 1, 5, 5);
  // AbstractLayer *hidden = new Layer(N, 1, 64, 1, 14, 14);
  AbstractLayer *output = new Layer(N, 1, labelcount, 1, 200, 1);
  // AbstractLayer *output = new Layer(N, 1, labelcount, 1, 14, 14);

  nn.add(input);
  nn.add(c1);
  nn.add(s1);
  nn.add(c2);
  nn.add(s2);
  // nn.add(c3);
  nn.add(hidden);
  nn.add(output);
  nn.train(0.05, 100);

  return 0;
}
