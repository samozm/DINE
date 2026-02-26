#define STRICT_R_HEADERS
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector>
#include <algorithm>
#include <random>
#include <typeinfo>
#include <deque>
#include <limits>
#include <cmath>
#include <omp.h>
// [[Rcpp::depends(RcppEigen)]]

double var(const Eigen::VectorXd & vec);

std::string printdims(const Eigen::MatrixXd & obj);

std::string printvec(const std::vector<double> & obj);

std::string BoolToString(bool b);

int BoolToInt(bool b);

int a_in_b(double a, const Eigen::VectorXd & b);

Eigen::ArrayXi loc_a_in_b(double a, const Eigen::VectorXd & b);

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

Rcpp::List calc_ZDZ_plus_E_list(const std::vector<Eigen::MatrixXd>& Z,
                          const Eigen::MatrixXd & D, const Eigen::VectorXd & E,
                          int n, int k, int t);

int make_MAP(const std::vector<Eigen::MatrixXd>& Z,
             Eigen::MatrixXd & masterZ, 
             Eigen::MatrixXi & MAP, Eigen::VectorXd & r0,
             int n, int k, int t);

void estimate_beta(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, 
                   const Eigen::VectorXi kt_vec, const Eigen::MatrixXi & MAP,
                   const std::vector<Eigen::MatrixXd> & V, Eigen::VectorXd & beta,
                   int n, int k, int t);

void estimate_beta2(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, 
                    const Eigen::MatrixXd & Z,
                    const Eigen::MatrixXd & D,
                    const Eigen::VectorXd & E,
                    const Eigen::VectorXi kt_vec, const Eigen::MatrixXi & MAP,
                    Eigen::VectorXd & beta,
                    int n, int k, int t);

Eigen::MatrixXd Et_assemble(const Eigen::VectorXd & E, 
                            const Eigen::MatrixXi & MAP, 
                            int i, int k, int t, int kt);

void Et_assemble_IP(const Eigen::VectorXd & E, 
                       Eigen::MatrixXd & Et,
                 const Eigen::MatrixXi & MAP, 
                 int i, int k, int t, int kt);

Eigen::MatrixXd Z_assemble(const Eigen::MatrixXd & masterZ, 
                           const Eigen::MatrixXi & MAP,
                           int i, int k, int t, int kt);

void Z_assemble_IP(const Eigen::MatrixXd & masterZ, 
                      Eigen::MatrixXd & Z_out,
                const Eigen::MatrixXi & MAP,
                int i, int k, int t, int kt);

Eigen::VectorXd R_expand(const Eigen::VectorXd & R,
                         const Eigen::MatrixXi & MAP,
                         int idx, int q);

Eigen::MatrixXd RtR(const Eigen::MatrixXd & R, const Eigen::MatrixXi & MAP);
