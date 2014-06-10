#include "AbstractLayer.h"

AbstractLayer::AbstractLayer() {
  m_type = N;
  m_plaincount = 0;
  m_height = 0;
  m_width = 0;
  m_pheight = 0;
  m_pwidth = 0;
  m_kernelcount = 0;
  m_kheight = 0;
  m_kwidth = 0;
  m_size = 0;
  m_a = NULL;
  m_beta = NULL;
  m_bias = NULL;
  m_delta = NULL;
  m_kernel = NULL;
  m_pbias = NULL;
  m_table = NULL;
  m_theta = NULL;
}
AbstractLayer::AbstractLayer(LayerType type, int plaincount, int height, int width, int pheight, int pwidth) {
  m_type = type;
  m_plaincount = plaincount;
  m_height = height;
  m_width = width;
  m_pheight = pheight;
  m_pwidth = pwidth;

  m_a = new double**[m_plaincount];
  m_bias = new double**[m_plaincount];
  m_delta = new double**[m_plaincount];
  for (int p = 0; p < m_plaincount; ++p) {
    m_a[p] = new double*[m_height];
    m_bias[p] = new double*[m_height];
    m_delta[p] = new double*[m_height];
    for (int i = 0; i < m_height; ++i) {
      m_a[p][i] = new double[m_width];
      m_bias[p][i] = new double[m_width];
      m_delta[p][i] = new double[m_width];
      for (int j = 0; j < m_width; ++j) {
        m_a[p][i][j] = 0.0;
        m_bias[p][i][j] = 0.0;
        m_delta[p][i][j] = 0.0;
      }
    }
  }
  m_pbias = new double[m_plaincount];
}
AbstractLayer::~AbstractLayer() {
  if (m_a) {
    for (int p = 0; p < m_plaincount; ++p) {
      for (int i = 0; i < m_height; ++i) {
        delete m_a[p][i];
      }
      delete m_a[p];
    }
    delete m_a;
    m_a = NULL;
  }
  if (m_bias) {
    for (int p = 0; p < m_plaincount; ++p) {
      for (int i = 0; i < m_height; ++i) {
        delete m_bias[p][i];
      }
      delete m_bias[p];
    }
    delete m_bias;
    m_bias = NULL;
  }
  if (m_delta) {
    for (int p = 0; p < m_plaincount; ++p) {
      for (int i = 0; i < m_height; ++i) {
        delete m_delta[p][i];
      }
      delete m_delta[p];
    }
    delete m_delta;
    m_delta = NULL;
  }
  delete m_pbias;
}

LayerType AbstractLayer::type(void) {
  return m_type;
}
int AbstractLayer::plain(void) {
  return m_plaincount;
}
int AbstractLayer::height(void) {
  return m_height;
}
int AbstractLayer::width(void) {
  return m_width;
}

void AbstractLayer::addConnection(int table[][2]) {}
void AbstractLayer::randomInitWeight(void) {
  for (int p = 0; p < m_plaincount; ++p) {
    for (int i = 0; i < m_height; ++i) {
      for (int j = 0; j < m_width; ++j) {
        m_bias[p][i][j] = randomX();
      }
    }
  }
}
void AbstractLayer::forward(AbstractLayer *&layer) {}
void AbstractLayer::backpropagate(AbstractLayer *&layer) {}
void AbstractLayer::update(double alpha, AbstractLayer *&layer) {}
void AbstractLayer::print(void) {}
