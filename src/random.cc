/*
 *  warca is a library for metric learning using weighted approximate rank
 *  component analysis algorithm written in c++.
 *
 *  Copyright (c) 2016 Idiap Research Institute, http://www.idiap.ch/
 *  Written by Cijo Jose <cijose@idiap.ch>
 *
 *  This file is part of warca.
 *
 *  warca is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License version 3 as
 *  published by the Free Software Foundation.
 *
 *  warca is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 *  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with selector.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "random.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

Prand::Prand(unsigned long s) {
  mt = new unsigned long[Prand::N];
  mt[0] = s & 0xffffffffUL;
  mti = 1;
  for (; mti < N; mti++) {
    mt[mti] = (1812433253UL * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + mti);
    /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
    /* In the previous versions, MSBs of the seed affect   */
    /* only MSBs of the array mt[].                        */
    /* 2002/01/09 modified by Makoto Matsumoto             */
    mt[mti] &= 0xffffffffUL;
    /* for >32 bit machines */
  }
  return_v = 0;
  v_val = 0.0;
}

Prand::Prand() {
  unsigned long s = 5489UL & (unsigned long)time(NULL);
  // printf("Hello world %d \n", Prand::N);
  mt = new unsigned long[Prand::N];
  mt[0] = s & 0xffffffffUL;
  mti = 1;
  for (; mti < N; mti++) {
    mt[mti] = (1812433253UL * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + mti);
    /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
    /* In the previous versions, MSBs of the seed affect   */
    /* only MSBs of the array mt[].                        */
    /* 2002/01/09 modified by Makoto Matsumoto             */
    mt[mti] &= 0xffffffffUL;
    /* for >32 bit machines */
  }
  return_v = 0;
  v_val = 0.0;
}

/* generates a random number on [0,0xffffffff]-interval */

unsigned long Prand::randi_32(void) {
  unsigned long y;
  static unsigned long mag01[2] = {0x0UL, MATRIX_A};
  /* mag01[x] = x * MATRIX_A  for x=0,1 */
  if (mti >= N) { /* generate N words at one time */
    int kk;
    for (kk = 0; kk < N - M; kk++) {
      y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
      mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }
    for (; kk < N - 1; kk++) {
      y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
      mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }
    y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
    mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1UL];
    mti = 0;
  }
  y = mt[mti++];
  /* Tempering */
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680UL;
  y ^= (y << 15) & 0xefc60000UL;
  y ^= (y >> 18);
  return y;
}

/* generates a random number on [0,0x7fffffff]-interval */

long Prand::randi_31(void) { return (long)(randi_32() >> 1); }
/* generates a random number on [0,1]-real-interval */

double Prand::rand_1(void) {
  return randi_32() * (1.0 / 4294967295.0);
  /* divided by 2^32-1 */
}
/* generates a random number on [0,1)-real-interval */

double Prand::rand_2(void) {
  return randi_32() * (1.0 / 4294967296.0);
  /* divided by 2^32 */
}
/* generates a random number on (0,1)-real-interval */

double Prand::rand_3(void) {
  return (((double)randi_32()) + 0.5) * (1.0 / 4294967296.0);
  /* divided by 2^32 */
}

/*
  Marsaglia polar method to generate normal random deviates
*/

double Prand::gauss_rng() {
  if (return_v) {
    return_v = 0;
    return v_val;
  }
  double u = 2 * rand_3() - 1;
  double v = 2 * rand_3() - 1;
  double r = u * u + v * v;
  if (r == 0 || r > 1)
    return gauss_rng();
  double c = sqrt(-2 * log(r) / r);
  v_val = v * c; // cache this for next
  return_v = 1;
  return u * c;
}
// Generate a random integer between a and b
int Prand::randi(int a, int b) {
  double rval = rand_1() * (double)(b - a) + (double)a;
  return floor(rval);
}
// Generate a floating point random variable  between a and b
double Prand::uniform_rng(double a, double b) { return rand_1() * (b - a) + a; }

// Generate a floating point random variable  between a and b
double Prand::randn(const double mu, const double stddev) {
  return gauss_rng() * stddev + mu;
}
