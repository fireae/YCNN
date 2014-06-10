/* Share.h --- Share
 */

#ifndef INCLUDED_SHARE_H
#define INCLUDED_SHARE_H 1

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <vector>

enum LayerType {
  N,
  L,
  C,
  S
};

const double SCALEWEIGHT = RAND_MAX;
const double epsilon  = 1e-1;
const double B = 2.0 / 3.0;
const double A = 1.7159;
const double MAX = RAND_MAX;

inline double sigmoid(double z) {
  // return 2.0 / (1.0 + exp(-z)) - 1.0;
  /*
  if (z + 132.0 <= epsilon) {
    return 0.0;
  }
  if (z - 132.0 >= epsilon) {
    return 1.0;
  }
  return 1.0 / (1.0 + exp(-z));
  */
  return tanh(B * z);
}
inline double linear(double z) {
  return z;
}
inline double df(double z) {
  return B * (1.0 - z * z);
  // return 1.0 - z * z;
  // return z * (1.0 - z);
}

inline double randomV(void) {
  static double v1, v2, s;
  static int phase = 0;
  double x;
  if (!phase) {
    do {
      double u1 = double(rand()) / RAND_MAX;
      double u2 = double(rand()) / RAND_MAX;
      v1 = 2 * u1 - 1;
      v2 = 2 * u2 - 1;
      s = v1 * v1 + v2 * v2;
    } while (s >= 1 || s == 0);
    x = v1 * sqrt(-2.0 * log(s) / s);
  }
  else {
    x = v2 * sqrt(-2.0 * log(s) / s);
  }
  phase ^= 1;
  return x * epsilon;
}
inline double randomX(void) {
  double ret = rand();
  return 2.0 * (ret / MAX) - 1.0;
}

#endif /* INCLUDED_SHARE_H */

