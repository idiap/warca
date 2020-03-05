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
#ifndef _RAND_H_
#define _RAND_H_

class Prand {
  static int const N = 624;
  static int const M = 397;
  static unsigned long const MATRIX_A = 0x9908b0dfUL; /* constant vector a */
  static unsigned long const UPPER_MASK =
      0x80000000UL; /* most significant w-r bits */
  static unsigned long const LOWER_MASK =
      0x7fffffffUL;  /* least significant r bits */
  int mti;           /* mti==N+1 means mt[N] is not initialized */
  unsigned long *mt; /* the array for the state vector  */
  bool return_v;
  double v_val;

public:
  Prand(unsigned long int seed);
  Prand();
  ~Prand() { delete[] mt; }
  // Generate gaussian random variable
  double gauss_rng();
  // Generate a floating point random variable  between a and b
  double uniform_rng(double a, double b);
  // Generate a integer random variable  between a and b
  int randi(int a, int b);
  // Generates a random number on [0,0xffffffff]-interval
  unsigned long int randi_32();
  // Generates a random number on [0,0x7fffffff]-interval
  long int randi_31();
  // Generate a random number on [0, 1] real interval
  double rand_1(void);
  // Generate a random number on [0, 1) real interval
  double rand_2(void);
  // Generate a random number on (0, 1) real interval
  double rand_3(void);
  // Generate a gaussian random variable with mean mu and standard deviation std
  double randn(const double mu, const double stddev);
};

#endif
