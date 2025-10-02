#define STRICT_R_HEADERS
#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector>
#include <algorithm>
#include <random>
#include <typeinfo>
#include <deque>
#include <limits>
// [[Rcpp::depends(RcppEigen)]]

std::string printdims(const Eigen::MatrixXd & obj);

std::string BoolToString(bool b);

int BoolToInt(bool b);

int a_in_b(double a, const Eigen::VectorXd & b);

void build_V_list_from_master(std::vector<Eigen::MatrixXd> & V, const Eigen::MatrixXd & master, const Eigen::MatrixXi & MAP, int n, int k, int t);


Eigen::MatrixXd covCalc(const Eigen::MatrixXd & X);

Eigen::MatrixXd covCalc(const Eigen::MatrixXd & X, const Eigen::MatrixXi & MAP, bool print=false);

void vec2list(const std::vector<Eigen::MatrixXd>& vec, Rcpp::List & out);

void list2vec(std::vector<Eigen::MatrixXd>& vec, const Rcpp::List & list);

void find_all(const std::vector<int> & vec, const int & val, std::deque<int> & out_val, std::deque<int> & out_not_val);

void find_all(const Eigen::VectorXi & vec, const int & val, std::deque<int> & out_val, std::deque<int> & out_not_val);

void calc_ZDZ_plus_E_list(const std::vector<Eigen::MatrixXd>& Z,
                          const Eigen::MatrixXd & D, const Eigen::VectorXd & E,
                          std::vector<Eigen::MatrixXd> & out, 
                          const Eigen::MatrixXi & MAP,
                          int n, int k, int t);

void estimate_beta(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, const Eigen::VectorXi kt_vec,
                   const std::vector<Eigen::MatrixXd> & V, Eigen::VectorXd & beta,
                   int n, int k, int t);

Eigen::MatrixXd Et_assemble(const Eigen::VectorXd & E, 
                            const Eigen::MatrixXi & MAP, 
                            int i, int k, int t, int kt);