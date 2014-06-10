#include "Layer.h"

Layer::Layer() {
  m_type = N;
  m_plaincount = 0;
  m_height = 0;
  m_width = 0;
  m_a = NULL;
  m_theta = NULL;
  m_bias = NULL;
  m_delta = NULL;
}
Layer::Layer(LayerType type, int plaincount, int height, int width, int pheight, int pwidth): AbstractLayer(type, plaincount, height, width, pheight, pwidth) {
  m_theta = new double****[m_plaincount];
  for (int p = 0; p < m_plaincount; ++p) {
    m_theta[p] = new double***[m_height];
    for (int i = 0; i < m_height; ++i) {
      m_theta[p][i] = new double**[m_width];
      for (int j = 0; j < m_width; ++j) {
        m_theta[p][i][j] = new double*[pheight];
        for (int pi = 0; pi < pheight; ++pi) {
          m_theta[p][i][j][pi] = new double[pwidth];
        }
      }
    }
  }
}
Layer::~Layer() {
  if (m_theta) {
    for (int p = 0; p < m_plaincount; ++p) {
      for (int i = 0; i < m_height; ++i) {
        for (int j = 0; j < m_width; ++j) {
          for (int pi = 0; pi < m_pheight; ++pi) {
            delete m_theta[p][i][j][pi];
          }
          delete m_theta[p][i][j];
        }
        delete m_theta[p][i];
      }
      delete m_theta[p];
    }
    delete m_theta;
    m_theta = NULL;
  }
}

void Layer::randomInitWeight(void) {
  for (int p = 0; p < m_plaincount; ++p) {
    for (int i = 0; i < m_height; ++i) {
      for (int j = 0; j < m_width; ++j) {
        m_bias[p][i][j] = randomX() * 0.3;
        for (int li = 0; li < m_pheight; ++li) {
          for (int lj = 0; lj < m_pwidth; ++lj) {
            m_theta[p][i][j][li][lj] = randomX() * 0.3;
          }
        }
      }
    }
  }
}
void Layer::forward(AbstractLayer *&layer) {
  int lheight = layer->height();
  int lwidth = layer->width();
  int lplaincount = layer->plain();
  for (int p = 0; p < m_plaincount; ++p) {
    for (int i = 0; i < m_height; ++i) {
      for (int j = 0; j < m_width; ++j) {
        double sum = m_bias[p][i][j];
        for (int pp = 0; pp < lplaincount; ++pp) {
          for (int li = 0; li < lheight; ++li) {
            for (int lj = 0; lj < lwidth; ++lj) {
              sum += m_theta[p][i][j][li][lj] * layer->m_a[pp][li][lj];
            }
          }
        }
        if (m_type == L) {
          m_a[p][i][j] = linear(sum);
        } else {
          m_a[p][i][j] = sigmoid(sum);
        }
      }
    }
  }
}
void Layer::backpropagate(AbstractLayer *&layer) {
  int lheight = layer->height();
  int lwidth = layer->width();
  int lplaincount = layer->plain();
  for (int p = 0; p < m_plaincount; ++p) {
    for (int i = 0; i < m_height; ++i) {
      for (int j = 0; j < m_width; ++j) {
        m_delta[p][i][j] = 0.0;
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

void Layer::update(double alpha, AbstractLayer *&layer) {
  int lheight = layer->height();
  int lwidth = layer->width();
  int lplaincount = layer->plain();
  for (int p = 0; p < m_plaincount; ++p) {
    for (int i = 0; i < m_height; ++i) {
      for (int j = 0; j < m_width; ++j) {
        m_bias[p][i][j] -= alpha * m_delta[p][i][j];
        for (int pp = 0; pp < lplaincount; ++pp) {
          for (int li = 0; li < lheight; ++li) {
            for (int lj = 0; lj < lwidth; ++lj) {
              m_theta[p][i][j][li][lj] -= alpha * m_delta[p][i][j] * layer->m_a[pp][li][lj];
            }
          }
        }
      }
    }
  }
}
void Layer::print(void) {
  for (int p = 0; p < m_plaincount; ++p) {
    printf("plain theta %d\n", p);
    for (int i = 0; i < m_height; ++i) {
      for (int j = 0; j < m_width; ++j) {
        printf("%lf ", m_theta[p][i][j][0][0]);
      }
      printf("\n");
    }
  }
  /*
  for (int p = 0; p < m_plaincount; ++p) {
    printf("plain delta %d\n", p);
    for (int i = 0; i < m_height; ++i) {
      for (int j = 0; j < m_width; ++j) {
        printf("%lf ", m_delta[p][i][j]);
      }
      printf("\n");
    }
  }
  for (int p = 0; p < m_plaincount; ++p) {
    printf("plain a %d\n", p);
    for (int i = 0; i < m_height; ++i) {
      for (int j = 0; j < m_width; ++j) {
        printf("%lf ", m_a[p][i][j]);
      }
      printf("\n");
    }
  }
  */
}
