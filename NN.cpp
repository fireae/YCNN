#include "NN.h"

#include <unistd.h>

NN::NN() {
  m_layercount = 0;
  m_layers.clear();
  m_x = NULL;
  m_y = NULL;
  m_xx = NULL;
}
NN::NN(int height, int width, int labelcount, int samplecount, int testcount) {
  m_layercount = 0;
  m_height = height;
  m_width = width;
  m_labelcount = labelcount;
  m_samplecount = samplecount;
  m_testcount = testcount;
  m_x = new double**[m_samplecount];
  for (int k = 0; k < m_samplecount; ++k) {
    m_x[k] = new double*[m_height];
    for (int i = 0; i < m_height; ++i) {
      m_x[k][i] = new double[m_width];
    }
  }
  m_y = new double*[m_samplecount];
  for (int k = 0; k < m_samplecount; ++k) {
    m_y[k] = new double[m_labelcount];
  }
  m_xx = new double**[m_testcount];
  for (int k = 0; k < m_testcount; ++k) {
    m_xx[k] = new double*[m_height];
    for (int i = 0; i < m_height; ++i) {
      m_xx[k][i] = new double[m_width];
    }
  }
}
NN::~NN() {
  if (m_x) {
    for (int k = 0; k < m_samplecount; ++k) {
      for (int i = 0; i < m_height; ++i) {
        delete m_x[k][i];
      }
      delete m_x[k];
    }
    delete m_x;
    m_x = NULL;
  }
  if (m_y) {
    for (int k = 0; k < m_samplecount; ++k) {
      delete m_y[k];
    }
    delete m_y;
    m_y = NULL;
  }
  if (m_xx) {
    for (int k = 0; k < m_testcount; ++k) {
      for (int i = 0; i < m_height; ++i) {
        delete m_xx[k][i];
      }
      delete m_xx[k];
    }
    delete m_xx;
    m_xx = NULL;
  }
}

void NN::add(AbstractLayer *&alayer) {
  ++m_layercount;
  m_layers.push_back(alayer);
}

void NN::loadTrainingDataFromCSV(char *s) {
  freopen(s, "r", stdin);
  for (int k = 0; k < m_samplecount; ++k) {
    fprintf(stderr, "\rLoading training data from csv... %.2lf%%", 100.0 * (k + 1) / m_samplecount);
    int id;
    int label;
    scanf("%d,%d", &id, &label);
    for (int i = 0; i < m_height; ++i) {
      for (int j = 0; j < m_width; ++j) {
        scanf(",%lf", &m_x[k][i][j]);
      }
    }
    for (int i = 0; i < m_labelcount; ++i) {
      m_y[k][i] = (i == label ? 1.0 : 0.0);
    }
  }
  fprintf(stderr, "\n");
}
void NN::loadTestDataFromCSV(char *s) {
  freopen(s, "r", stdin);
  for (int k = 0; k < m_testcount; ++k) {
    fprintf(stderr, "\rLoading test data from csv... %.2lf%%", 100.0 * (k + 1) / m_testcount);
    int id;
    scanf("%d", &id);
    for (int i = 0; i < m_height; ++i) {
      for (int j = 0; j < m_width; ++j) {
        scanf(",%lf", &m_xx[k][i][j]);
      }
    }
  }
  fprintf(stderr, "\n");
}

void NN::randomInitWeight(void) {
  srand(time(NULL));
  for (int l = 1; l < m_layercount; ++l) {
    m_layers[l]->randomInitWeight();
  }
  printf("Init weight done...\n");
}

void NN::forward(int k) {
  for (int i = 0; i < m_height; ++i) {
    for (int j = 0; j < m_width; ++j) {
      m_layers[0]->m_a[0][i][j] = m_x[k][i][j];
    }
  }
  for (int l = 1; l < m_layercount; ++l) {
    m_layers[l]->forward(m_layers[l - 1]);
  }
}
void NN::backpropagate(int k) {
  for (int i = 0; i < m_labelcount; ++i) {
    m_layers[m_layercount - 1]->m_delta[0][i][0] = (m_layers[m_layercount - 1]->m_a[0][i][0] - m_y[k][i]) * df(m_layers[m_layercount - 1]->m_a[0][i][0]);
  }
  for (int l = m_layercount - 2; l > 0; --l) {
    m_layers[l]->backpropagate(m_layers[l + 1]);
  }
}
void NN::update(double alpha) {
  for (int l = 1; l < m_layercount; ++l) {
    m_layers[l]->update(alpha, m_layers[l - 1]);
  }
}

void NN::check(int k) {
  forward(k);
  for (int i = 0; i < m_labelcount; ++i) {
    m_layers[m_layercount - 1]->m_delta[0][i][0] = (m_layers[m_layercount - 1]->m_a[0][i][0] - m_y[k][i]) * df(m_layers[m_layercount - 1]->m_a[0][i][0]);
  }
  for (int i = 0; i < m_labelcount; ++i) {
    printf("%lf %lf %lf\n", m_layers[m_layercount - 1]->m_a[0][i][0], m_y[k][i], m_layers[m_layercount - 1]->m_delta[0][i][0]);
  }
}

int NN::validate(void) {
  int ret = 0;
  for (int k = 0; k < 1000; ++k) {
    forward(k);
    double maxpro = -2.0;
    int label;
    for (int i = 0; i < m_labelcount; ++i) {
      if (m_layers[m_layercount - 1]->m_a[0][i][0] > maxpro) {
        maxpro = m_layers[m_layercount - 1]->m_a[0][i][0];
        label = i;
      }
    }
    if (fabs(m_y[k][label] - 1.0) <= 0.01) {
      ++ret;
    }
  }
  return ret;
}

void NN::validate(double &error) {
}

void NN::applyResult(void) {
}

void NN::train(double alpha, int step) {
  int counter = 0;
  int best = 0;
  randomInitWeight();
  printf("Begin...\n");
  while (true) {
    ++counter;
    for (int k = 1000; k < m_samplecount; ++k) {
      fprintf(stderr, "\rRate of progress: %.2lf%%", 100.0 * (double)(k + 1) / (double)m_samplecount);
      forward(k);
      backpropagate(k);
      update(alpha);
      // m_layers[3]->print();
    }
    if (counter % 10 == 0) {
      alpha *= 0.9;
    }
    fprintf(stderr, "\n");
    int valid = validate();
    if (valid > best) {
      best = valid;
    }
    printf("time %d, validate: %d, best: %d\n", counter, valid, best);
    check(0);
    printf("\n");
  }
}
