#include "ConvolutionLayer.h"

ConvolutionLayer::ConvolutionLayer(): AbstractLayer() {
  m_type = C;
}
ConvolutionLayer::ConvolutionLayer(int plaincount, int height, int width, int pheight, int pwidth, int kernelcount, int kheight, int kwidth): AbstractLayer(C, plaincount, height, width, pheight, pwidth) {
  m_kernelcount = kernelcount;
  m_kheight = kheight;
  m_kwidth = kwidth;
  m_kernel = new double**[m_kernelcount];
  for (int kc = 0; kc < m_kernelcount; ++kc) {
    m_kernel[kc] = new double*[m_kheight];
    for (int i = 0; i < m_kheight; ++i) {
      m_kernel[kc][i] = new double[m_kwidth];
    }
  }
  m_table = new std::vector<int>[m_kernelcount];
}
ConvolutionLayer::~ConvolutionLayer() {
  if (m_kernel) {
    for (int kc = 0; kc < m_kernelcount; ++kc) {
      for (int i = 0; i < m_kheight; ++i) {
        delete m_kernel[kc][i];
      }
      delete m_kernel[kc];
    }
    delete m_kernel;
    m_kernel = NULL;
  }
  if (m_table) {
    delete m_table;
  }
}

int ConvolutionLayer::kernelcount(void) {
  return m_kernelcount;
}
int ConvolutionLayer::kheight(void) {
  return m_kheight;
}
int ConvolutionLayer::kwidth(void) {
  return m_kwidth;
}

void ConvolutionLayer::addConnection(int table[][2]) {
  for (int i = 0; i < m_kernelcount; ++i) {
    for (int j = 0; j < 2; ++j) {
      m_table[i].push_back(table[i][j]);
    }
  }
}
void ConvolutionLayer::randomInitWeight(void) {
  for (int p = 0; p < m_plaincount; ++p) {
    m_pbias[p] = randomX() * 0.3;
  }
  for (int kc = 0; kc < m_kernelcount; ++kc) {
    for (int i = 0; i < m_kheight; ++i) {
      for (int j = 0; j < m_kwidth; ++j) {
        m_kernel[kc][i][j] = randomX() * 0.3;
      }
    }
  }
}
void ConvolutionLayer::forward(AbstractLayer *&layer) {
  for (int p = 0; p < m_plaincount; ++p) {
    for (int i = 0; i < m_height; ++i) {
      for (int j = 0; j < m_width; ++j) {
        m_a[p][i][j] = 0.0;
      }
    }
  }
  double **kernel180;
  kernel180 = new double*[m_kheight];
  for (int i = 0; i < m_kheight; ++i) {
    kernel180[i] = new double[m_kwidth];
  }
  for (int kc = 0; kc < m_kernelcount; ++kc) {
    int p = m_table[kc][0]; // current plain
    int pp = m_table[kc][1]; // previous plain
    for (int i = 0; i < m_kheight; ++i) {
      for (int j = 0; j < m_kwidth; ++j) {
        kernel180[i][j] = m_kernel[kc][m_kheight - 1 - i][m_kwidth - 1 - j];
      }
    }
    for (int i = 0; i < m_height; ++i) {
      for (int j = 0; j < m_width; ++j) {
        // convolution
        double sum = 0.0;
        for (int ii = 0; ii < m_kheight; ++ii) {
          for (int jj = 0; jj < m_kwidth; ++jj) {
            sum += kernel180[ii][jj] * layer->m_a[pp][i + ii][j + jj];
          }
        }
        m_a[p][i][j] += sum;
      }
    }
  }
  for (int p = 0; p < m_plaincount; ++p) {
    for (int i = 0; i < m_height; ++i) {
      for (int j = 0; j < m_width; ++j) {
        m_a[p][i][j] = sigmoid(m_a[p][i][j] + m_pbias[p]);
      }
    }
  }
  if (kernel180) {
    for (int i = 0; i < m_kheight; ++i) {
      delete kernel180[i];
    }
    delete kernel180;
  }
}
void ConvolutionLayer::backpropagate(AbstractLayer *&layer) {
  for (int p = 0; p < m_plaincount; ++p) {
    for (int i = 0; i < m_height; ++i) {
      for (int j = 0; j < m_width; ++j) {
        m_delta[p][i][j] = 0.0;
      }
    }
  }
  if (layer->type() == S) {
    int size = layer->m_size;
    for (int p = 0; p < m_plaincount; ++p) {
      for (int i = 0; i < m_height; ++i) {
        for (int j = 0; j < m_width; ++j) {
          // m_delta[p][i][j] = layer->m_beta[p] * df(m_a[p][i][j]) * (layer->m_delta[p][i / size][j / size] / (double)(size * size));
          m_delta[p][i][j] = df(m_a[p][i][j]) * (layer->m_delta[p][i / size][j / size] / (double)(size * size));
        }
      }
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
void ConvolutionLayer::update(double alpha, AbstractLayer *&layer) {
  for (int p = 0; p < m_plaincount; ++p) {
    for (int i = 0; i < m_height; ++i) {
      for (int j = 0; j < m_width; ++j) {
        m_pbias[p] -= alpha * m_delta[p][i][j];
      }
    }
  }
  for (int kc = 0; kc < m_kernelcount; ++kc) {
    int p = m_table[kc][0]; // current plain
    int pp = m_table[kc][1];
    for (int i = 0; i < m_height; ++i) {
      for (int j = 0; j < m_width; ++j) {
        for (int ii = 0; ii < m_kheight; ++ii) {
          for (int jj = 0; jj < m_kwidth; ++jj) {
            m_kernel[kc][m_kheight - 1 - ii][m_kwidth - 1 - jj] -= alpha * m_delta[p][i][j] * layer->m_a[pp][i + ii][j + jj];
          }
        }
      }
    }
  }
}
void ConvolutionLayer::print(void) {
  for (int kc = 0; kc < m_kernelcount; ++kc) {
    printf("kernel %d\n", kc);
    for (int i = 0; i < m_kheight; ++i) {
      for (int j = 0; j < m_kwidth; ++j) {
        printf("%lf ", m_kernel[kc][i][j]);
      }
      printf("\n");
    }
  }
  for (int p = 0; p < m_plaincount; ++p) {
    printf("plain %d bias: %lf\n", p, m_pbias[p]);
  }
  for (int kc = 0; kc < m_kernelcount; ++kc) {
    printf("kernel %d, table %d, %d\n", kc, m_table[kc][0], m_table[kc][1]);
  }
  for (int p = 0; p < m_plaincount; ++p) {
    printf("plain %d\n", p);
    for (int i = 0; i < m_height; ++i) {
      for (int j = 0; j < m_width; ++j) {
        printf("%lf ", m_a[p][i][j]);
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("\n");
}
