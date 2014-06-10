#include "SamplingLayer.h"

SamplingLayer::SamplingLayer(): AbstractLayer() {
  m_type = S;
}
SamplingLayer::SamplingLayer(int plaincount, int height, int width, int pheight, int pwidth, int size): AbstractLayer(S, plaincount, height, width, pheight, pwidth) {
  m_size = size;
  m_beta = new double[m_plaincount];
}
SamplingLayer::~SamplingLayer() {
  if (m_beta) {
    delete m_beta;
    m_beta = NULL;
  }
}

void SamplingLayer::randomInitWeight(void) {
  for (int p = 0; p < m_plaincount; ++p) {
    m_beta[p] = randomX();
    m_pbias[p] = randomX();
  }
}
void SamplingLayer::forward(AbstractLayer *&layer) {
  double factor = (double)(m_size * m_size);
  for (int p = 0; p < m_plaincount; ++p) {
    for (int i = 0; i < m_height; ++i) {
      for (int j = 0; j < m_width; ++j) {
        int ph = i * m_size;
        int pw = j * m_size;
        double sum = 0.0;
        for (int ii = 0; ii < m_size; ++ii) {
          for (int jj = 0; jj < m_size; ++jj) {
            sum += layer->m_a[p][ph + ii][pw + jj] * factor;
          }
        }
        // sum *= m_beta[p];
        m_a[p][i][j] = sigmoid(sum + m_pbias[p]);
      }
    }
  }
}
void SamplingLayer::backpropagate(AbstractLayer *&layer) {
  for (int p = 0; p < m_plaincount; ++p) {
    for (int i = 0; i < m_height; ++i) {
      for (int j = 0; j < m_width; ++j) {
        m_delta[p][i][j] = 0.0;
      }
    }
  }
  if (layer->type() == C) {
    int kernelcount = layer->m_kernelcount;
    int kheight = layer->m_kheight;
    int kwidth = layer->m_kwidth;
    int pheight = layer->height();
    int pwidth = layer->width();

    int padding_height = m_height + kheight - 1;
    int padding_width = m_width + kwidth - 1;
    double **padding;
    double **kernel180;
    padding = new double*[padding_height];
    for (int i = 0; i < padding_height; ++i) {
      padding[i] = new double[padding_width];
      for (int j = 0; j < padding_width; ++j) {
        padding[i][j] = 0.0;
      }
    }
    kernel180 = new double*[kheight];
    for (int i = 0; i < kheight; ++i) {
      kernel180[i] = new double[kwidth];
    }

    for (int kc = 0; kc < kernelcount; ++kc) {
      int p = layer->m_table[kc][0];
      int pp = layer->m_table[kc][1];
      for (int i = 0; i < pheight; ++i) {
        for (int j = 0; j < pwidth; ++j) {
          padding[i + kheight - 1][j + kwidth - 1] = layer->m_delta[p][i][j];
        }
      }
      for (int i = 0; i < kheight; ++i) {
        for (int j = 0; j < kwidth; ++j) {
          kernel180[i][j] = layer->m_kernel[kc][kheight - 1 - i][kwidth - 1 - j];
        }
      }
      for (int i = 0; i < m_height; ++i) {
        for (int j = 0; j < m_width; ++j) {
          double sum = 0.0;
          for (int ii = 0; ii < kheight; ++ii) {
            for (int jj = 0; jj < kwidth; ++jj) {
              sum += kernel180[ii][jj] * padding[i + ii][j + jj];
            }
          }
          m_delta[pp][i][j] += sum * df(m_a[pp][i][j]);
        }
      }
    }
    if (padding) {
      for (int i = 0; i < padding_height; ++i) {
        delete padding[i];
      }
      delete padding;
    }
    if (kernel180) {
      for (int i = 0; i < kheight; ++i) {
        delete kernel180[i];
      }
      delete kernel180;
    }
  } else {
    int lheight = layer->height();
    int lwidth = layer->width();
    int lplaincount = layer->plain();
    for (int p = 0; p < m_plaincount; ++p) {
      for (int i = 0; i < m_height; ++i) {
        for (int j = 0; j < m_width; ++j) {
          for (int pp = 0; pp < lplaincount; ++pp) {
            for (int li = 0; li < lheight; ++li) {
              for (int lj = 0; lj < lwidth; ++lj) {
                m_delta[p][i][j] += layer->m_theta[pp][li][lj][i][j] * layer->m_delta[pp][li][lj];
              }
            }
          }
          m_delta[p][i][j] *= df(m_a[p][i][j]);
        }
      }
    }
  }
}
void SamplingLayer::update(double alpha, AbstractLayer *&layer) {
  for (int p = 0; p < m_plaincount; ++p) {
    for (int i = 0; i < m_height; ++i) {
      for (int j = 0; j < m_width; ++j) {
        m_pbias[p] -= alpha * m_delta[p][i][j];
        int ph = i * m_size;
        int pw = j * m_size;
        double sum = 0.0;
        for (int ii = 0; ii < m_size; ++ii) {
          for (int jj = 0; jj < m_size; ++jj) {
            sum += layer->m_a[p][i + ii][j + jj];
          }
        }
        m_beta[p] -= alpha * (sum / (double)(m_size * m_size));
      }
    }
  }
}
void SamplingLayer::print(void) {
  for (int p = 0; p < m_plaincount; ++p) {
    printf("p %d\n", p);
    printf("beta %lf, bias %lf\n", m_beta[p], m_pbias[p]);
    for (int i = 0; i < m_height; ++i) {
      for (int j = 0; j < m_width; ++j) {
        printf("%lf ", m_a[p][i][j]);
      }
      printf("\n");
    }
  }
  printf("\n");
}
