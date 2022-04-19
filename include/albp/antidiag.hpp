
#pragma once

#include <string>
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace albp {

int antidiag(
  const std::string &seqa,
  const std::string &seqb,
  const int gap_open,
  const int gap_extend,
  const int match_score,
  const int mismatch_score,
  int* max_x = NULL, int* max_y = NULL
);

}