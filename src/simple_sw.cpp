#include <albp/simple_sw.hpp>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>



namespace albp {

SimpleSmithWatermanResult simple_smith_waterman(
  const std::string &seqa,
  const std::string &seqb,
  const int gap_open,
  const int gap_extend,
  const int match_score,
  const int mismatch_score,
  int* max_x = NULL, int* max_y = NULL
){
  SimpleSmithWatermanResult ret;
  ret.size = seqa.size()*seqb.size();
  ret.E = Matrix2D<int>(seqa.size(),seqb.size());
  ret.F = Matrix2D<int>(seqa.size(),seqb.size());
  ret.H = Matrix2D<int>(seqa.size(),seqb.size());
  ret.seqa = seqa;
  ret.seqb = seqb;
  int m_x = -1;
  int m_y = -1;
  ret.score = 0;

  const auto mm = [&](const char a, const char b){ return (a==b)?match_score:mismatch_score; };

  for(size_t i=0;i<seqa.size();i++)
  for(size_t j=0;j<seqb.size();j++){
    ret.E(i,j) = std::max(
      (j>0) ? ret.E(i,j-1) + gap_extend : 0,
      (j>0) ? ret.H(i,j-1) + gap_open   : 0
    );

    ret.F(i,j) = std::max(
      (i>0) ? ret.F(i-1,j) + gap_extend : 0,
      (i>0) ? ret.H(i-1,j) + gap_open   : 0
    );

    ret.H(i,j) = std::max(
      std::max(0, ret.E(i,j)),
      std::max(
        ret.F(i,j),
        ((i>0 && j>0) ? ret.H(i-1,j-1) : 0) + mm(seqa[i], seqb[j])
      )
    );
    if (ret.H(i,j) > ret.score)
    {
      ret.score = ret.H(i,j);
      m_x = i;
      m_y = j;
    }
  }

  ret.score = ret.H.max();
  if(max_x != NULL && max_y != NULL)
  {
    *max_x = m_x;
    *max_y = m_y;
  }

  return ret;
}

}