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
#include "utils.h"
// [[Rcpp::depends(RcppEigen)]]

double var(const Eigen::VectorXd & vec)
{
    return((vec.array() - vec.array().mean()).square().mean());
}

std::string printdims(const Eigen::MatrixXd & obj)
{
    return ("("+ std::to_string(obj.rows()) + ", " + std::to_string(obj.cols()) + ")\n");
}

std::string printvec(const std::vector<double> & obj)
{
  std::string out = std::to_string(obj[0]);
  for(int i=1; i<obj.size(); ++i)
  {
    out.append(", ").append(std::to_string(obj[i]));
  }
  return(out);
}

std::string BoolToString(bool b)
{
  return b ? "true" : "false";
}

int BoolToInt(bool b)
{
    return b ? 1 : 0;
}

int a_in_b(double a, const Eigen::VectorXd & b)
{
    for(double i: b)
    {
        if(std::abs(a-i)<0.01*b.cwiseAbs().maxCoeff())  //std::max(std::abs(b)))
        {
            return(1);
        }
    }
    return(0);
}

Eigen::ArrayXi loc_a_in_b(double a, const Eigen::VectorXd & b)
{
    std::deque<int> out;
    int n_idx = 0;
    for(int i=0; i<b.rows(); ++i)
    {
        if(std::abs(a-b(i))<0.01*b.cwiseAbs().maxCoeff())  //std::max(std::abs(b)))
        {
            out.push_back(i);
            n_idx++;
        }
    }
    Eigen::ArrayXi out_vec(n_idx);
    for(int i=0; i<n_idx; ++i)
    {
        out_vec(i) = out[i];
    }
    return(out_vec);
}

void build_V_list_from_master(std::vector<Eigen::MatrixXd> & V, const Eigen::MatrixXd & master, const Eigen::MatrixXi & MAP, int n, int k, int t)
{
  for(int i=0; i<n; ++i)
  {
    int kt0 = MAP.rowwise().sum()(i);
    V[i] = Eigen::MatrixXd::Zero(kt0,kt0);
    int cnt0 = 0; 
    for(int j=0; j<(k*t);++j)
    {
        if(MAP(i,j) == 1)
        {
            int cnt1 = 0;
            for(int l=0; l<(k*t);++l)
            {
                if(MAP(i,l) == 1)
                {
                    V[i](cnt0,cnt1) = master(j,l);
                    cnt1++;
                }
            }
            cnt0++;
        }
    }
  }
}


Eigen::MatrixXd covCalc(const Eigen::MatrixXd & X)
{
    Eigen::MatrixXd centeredX = X.rowwise() - X.colwise().mean();
    Eigen::MatrixXd cov = (centeredX.adjoint() * centeredX) / double(X.rows());
    return(cov);
}

// [[Rcpp::export]]
Eigen::MatrixXd covCalc(const Eigen::MatrixXd & X, const Eigen::MatrixXi & MAP, bool print)
{
    int n = X.rows();
    int p = X.cols();

    // Select values of X where MAP is positive, otherwise input 0
    Eigen::MatrixXd Xz = (MAP.array() != 0).select(X,0.0);

    Eigen::MatrixXd MAPd = MAP.cast<double>();
    // Sum of Xi * Xj for all pairs of nodes
    Eigen::MatrixXd SumXY = Xz.transpose() * Xz;
    // number of observations shared by i and j
    Eigen::MatrixXd N = MAPd.transpose() * MAPd;
    // Sum of X(i,) where MAP(i,j) is 1
    Eigen::MatrixXd SumX_shared = Xz.transpose() * MAPd;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(p,p);

    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i < p; ++i)
    {
        for(int j=0; j <= i; ++j)
        {
            double Nij = N(i,j);
            if(Nij > 1.0)
            {
                double val = (SumXY(i, j) - (SumX_shared(i, j) * SumX_shared(j, i) / Nij)) / Nij;
                cov(i,j) = val;
                if(i != j) 
                {
                    cov(j,i) = val;
                }
            }
        }
    }
    return(cov);
}

Eigen::VectorXd varCalcDiag(const Eigen::MatrixXd & X, const Eigen::MatrixXi & MAP, bool print)
{
    int n = X.rows();
    int p = X.cols();
    Eigen::VectorXd var = Eigen::VectorXd::Zero(p);

    for(int j = 0; j < p; ++j)
    {
        double sum = 0.0;
        double sumSq = 0.0;
        int n_obs = 0;
        for(int i=0;i<n;++i)
        {
            if(MAP(i,j) != 0)
            {
                double tmp = X(i,j);
                sum += tmp;
                sumSq += tmp * tmp;
                n_obs++;
            }
        }
        if(n_obs > 1)
        {
            double mean = sum/n_obs;
            var(j) = (sumSq / n_obs) - (mean * mean);
        }
    }
    return(var);
}

void vec2list(const std::vector<Eigen::MatrixXd>& vec, Rcpp::List & out)
{
    for(int i=0;i<vec.size();++i) //(Eigen::MatrixXd i:vec)
    {
        out[i] = vec[i];//out.push_back(i);
    }
}

void list2vec(std::vector<Eigen::MatrixXd>& vec, const Rcpp::List & list)
{
    if(list.size() != vec.size())
    {
        Rcpp::Rcout << "VEC SIZE (" << vec.size() << ") INCOMPATIBLE W/ LIST SIZE (" << list.size() << ")\n"; 
    }
    for(int i=0;i<list.size();++i) //(Eigen::MatrixXd i:vec)
    {
        vec[i] = list[i];//out.push_back(i);
    }
}

void find_all(const std::vector<int> & vec, const int & val, std::deque<int> & out_val, std::deque<int> & out_not_val)
{
    for(int iter=0;iter<vec.size();++iter)
    {
        if(vec[iter] == val)
        {
            out_val.push_back(iter);
        } else
        {
            out_not_val.push_back(iter);
        }
    }
}

void find_all(const Eigen::VectorXi & vec, const int & val, std::deque<int> & out_val, std::deque<int> & out_not_val)
{
    for(int iter=0;iter<vec.size();++iter)
    {
        if(vec(iter) == val)
        {
            out_val.push_back(iter);
        } else
        {
            out_not_val.push_back(iter);
        }
    }
}

int make_MAP(const std::vector<Eigen::MatrixXd>& Z, 
             Eigen::MatrixXd & masterZ,
             Eigen::MatrixXi & MAP, Eigen::VectorXd & r0,
             int n, int k, int t)
{
    std::deque<double> all_times; //= Z(Eigen::all,Eigen::seqN(1,2*k,2)).reshaped();
    int nkt = 0;
    for(int i=0; i<n; ++i)
    {
        nkt += Z[i].rows();
        int zcol = Z[i].cols();
        for(int l=0; l<k; ++l)
        {
            Eigen::ArrayXi idxs = loc_a_in_b(1,Z[i](Eigen::all,2*l));
            Eigen::VectorXd n_times = Z[i](idxs,2*l+1).reshaped();
            for(int j=0;j<n_times.size();++j)
            {
                all_times.push_back(n_times(j));
            }
        }
    }
    r0 = Eigen::VectorXd::Zero(nkt);
    std::set<double> time_set{all_times.begin(), all_times.end()};
    std::vector<double> times(time_set.begin(), time_set.end());
    MAP = Eigen::MatrixXi::Zero(n,k*t);
    masterZ = Eigen::MatrixXd::Zero(k*t,2*k);
    for(int i = 0; i < n; ++i)
    {
        for(int l=0; l<k; ++l)
        {
            //TODO: ENSURE CORRECTLY DISTINGUISHING BETWEEN TIME 0 and NO DATA 0
            for(int j=0; j < t; ++j)
            {
                if(times[j] == 0)
                {
                    Eigen::ArrayXi idxs = loc_a_in_b(0,Z[i](Eigen::all,2*l + 1));
                    MAP(i,l*t + j) = a_in_b(1,Z[i](idxs,2*l));
                } else {
                    MAP(i,l*t + j) = a_in_b(times[j],Z[i](Eigen::all,2*l + 1));
                }
            }
        }
    }
    int CELL = 0;
    for(int l=0; l<k; ++l)
    {
        masterZ(Eigen::seqN(l*t,t),CELL) = Eigen::VectorXd::Constant(t,1.0);
        masterZ(Eigen::seqN(l*t,t),CELL+1) = Eigen::Map<Eigen::VectorXd>(times.data(), t);
        CELL += 2;
    }
    return(nkt);
}

Eigen::MatrixXd Et_assemble(const Eigen::VectorXd & E, 
                            const Eigen::MatrixXi & MAP, 
                            int i, int k, int t, int kt)
{
  Eigen::MatrixXd Et = Eigen::MatrixXd::Zero(kt,kt);
  int cnt = 0;
  int cnt2 = 0;
  for(int j = 0; j < k; ++j)
  {
    for(int l = 0; l < t; ++l)
    {
      if(MAP(i,cnt2) == 1)
      {
        Et(cnt,cnt) = E(j);
        cnt++;
      }
      cnt2++;
    }
  }
  return(Et);
}

Eigen::MatrixXd Z_assemble(const Eigen::MatrixXd & masterZ, 
                           const Eigen::MatrixXi & MAP,
                           int i, int k, int t, int kt)
{
    Eigen::MatrixXd Z_out = Eigen::MatrixXd::Zero(kt,2*k);
    int cnt = 0;
    int cnt2 = 0;
    for(int j = 0; j<k; ++j)
    {
        for(int l = 0; l < t; ++l)
        {
            if(MAP(i,cnt2) == 1)
            {
                Z_out(cnt,Eigen::all) = masterZ(cnt2,Eigen::all);
                cnt++;
            }
            cnt2++;
        }
    }
    return(Z_out);
}

void Et_assemble_IP(const Eigen::VectorXd & E, 
                       Eigen::MatrixXd & Et,
                 const Eigen::MatrixXi & MAP, 
                 int i, int k, int t, int kt)
{
    Et.setZero(kt,kt);
    int cnt = 0;
    int cnt2 = 0;
    for(int j = 0; j < k; ++j)
    {
        for(int l = 0; l < t; ++l)
        {
        if(MAP(i,cnt2) == 1)
        {
            Et.diagonal()(cnt) = E(j);
            cnt++;
        }
        cnt2++;
        }
    }
}

void Z_assemble_IP(const Eigen::MatrixXd & masterZ, 
                      Eigen::MatrixXd & Z_out,
                const Eigen::MatrixXi & MAP,
                int i, int k, int t, int kt)
{
    Z_out.resize(kt,2*k);
    int cnt = 0;
    for(int j = 0; j<k*t; ++j)
    {
        if(MAP(i,j) == 1)
        {
            Z_out.row(cnt).noalias() = masterZ.row(j);
            cnt++;
        }
    }
}

void calc_ZDZ_plus_E_list(const std::vector<Eigen::MatrixXd>& Z,
                          const Eigen::MatrixXd & D, const Eigen::VectorXd & E,
                          std::vector<Eigen::MatrixXd> & out, 
                          const Eigen::MatrixXi & MAP,
                          int n, int k, int t)
{
    for(int i=0; i<n; ++i)
    {
        int kt = Z[i].rows();
        out[i] = (Z[i] * D * Z[i].transpose() + Et_assemble(E, MAP, i, k, t, kt));
    }
}

// [[Rcpp::export]]
Rcpp::List calc_ZDZ_plus_E_list(const std::vector<Eigen::MatrixXd>& Z,
                          const Eigen::MatrixXd & D, const Eigen::VectorXd & E,
                          int n, int k, int t)
{
    Eigen::MatrixXi MAP = Eigen::MatrixXi::Zero(n,k*t);
    Eigen::VectorXd r0;
    Eigen::MatrixXd masterZ;
    make_MAP(Z,masterZ,MAP,r0,n,k,t);
    std::vector<Eigen::MatrixXd> out(n);
    calc_ZDZ_plus_E_list(Z,D,E,out,MAP,n,k,t);
    return(Rcpp::wrap(out));
}



void estimate_beta(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, 
                   const Eigen::VectorXi kt_vec, const Eigen::MatrixXi & MAP,
                   const std::vector<Eigen::MatrixXd> & V, Eigen::VectorXd & beta,
                   int n, int k, int t)
{
    int q = X.cols();
    Eigen::MatrixXd XVX = Eigen::MatrixXd::Zero(q,q);
    int cnt = 0;
    std::vector<bool> v_inv(n);
    std::vector<Eigen::MatrixXd> V_inv(n);
    for(int i=0;i<n;++i)
    {
        int kt = kt_vec(i);
        Eigen::FullPivLU<Eigen::MatrixXd> lu(V[i]);
        v_inv[i] = lu.isInvertible();
        if(v_inv[i])
        {
            XVX += X(Eigen::seqN(cnt,kt),Eigen::all).transpose() * V[i].colPivHouseholderQr().solve(X(Eigen::seqN(cnt,kt),Eigen::all));
        }
        else
        {
            //Rcpp::Rcout << "Singular matrix - using pseudoinv" << "\n";
            //Rcpp::Rcout << "idx = " << idx << " sjt= " << i << "\n";
            //Rcpp::Rcout << "n0 = " << n << "; k0= " << k << "; t0=" << t << "\n";
            V_inv[i] = V[i].completeOrthogonalDecomposition().pseudoInverse();
            XVX += X(Eigen::seqN(cnt,kt),Eigen::all).transpose() * V_inv[i] * X(Eigen::seqN(cnt,kt),Eigen::all);
        }
        cnt += kt;
    }
    Eigen::MatrixXd XVXinvXt;
    Eigen::FullPivLU<Eigen::MatrixXd> lu(XVX);
    bool v_inv2 = lu.isInvertible();
    if(v_inv2)
    {
        XVXinvXt = XVX.colPivHouseholderQr().solve(X.transpose());
    }
    else
    {
        //Rcpp::Rcout << "Singular matrix - using pseudoinv" << "\n";
        XVXinvXt = XVX.completeOrthogonalDecomposition().pseudoInverse() * X.transpose();
    }
    beta = Eigen::VectorXd::Zero(q);
    cnt = 0;
    for(int i=0;i<n;++i)
    {
        int kt = kt_vec(i);
        if(v_inv[i])
        {
            beta += XVXinvXt(Eigen::all,Eigen::seqN(cnt,kt)) * V[i].colPivHouseholderQr().solve(y(Eigen::seqN(cnt,kt)));
        }
        else
        {
            beta += XVXinvXt(Eigen::all,Eigen::seqN(cnt,kt)) * V_inv[i] * y(Eigen::seqN(cnt,kt));
        }
        cnt += kt;
    }
}

void estimate_beta2(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, 
                    const Eigen::MatrixXd & Z,
                    const Eigen::MatrixXd & D,
                    const Eigen::VectorXd & E,
                    const Eigen::VectorXi kt_vec, const Eigen::MatrixXi & MAP,
                    Eigen::VectorXd & beta,
                    int n, int k, int t)
{
    int q = X.cols();
    Eigen::MatrixXd XVX = Eigen::MatrixXd::Zero(q,q);
    Eigen::VectorXd XVy = Eigen::VectorXd::Zero(q);
    Eigen::MatrixXd Zi, Et, ZiD, V, Xi;
    Eigen::VectorXd yi;
    int cnt = 0;
    
    for(int i=0;i<n;++i)
    {
        int kt = kt_vec(i);
        Z_assemble_IP(Z,Zi,MAP,i,k,t,kt);
        Et_assemble_IP(E,Et,MAP,i,k,t,kt);
        ZiD.resize(kt, 2*k);
        ZiD.noalias() = Zi * D;

        V.resize(kt, kt);
        V.noalias() = ZiD * Zi.transpose();
        V += Et;

        // Map block segments for X and y
        Xi = X.block(cnt, 0, kt, q);
        yi = y.segment(cnt, kt);


        Eigen::LDLT<Eigen::MatrixXd> ldlt(V);
        if(ldlt.info() == Eigen::Success)
        {
            Eigen::MatrixXd VinvXi = ldlt.solve(Xi);
            Eigen::VectorXd Vinvyi = ldlt.solve(yi);
            
            XVX.noalias() += Xi.transpose() * VinvXi;
            XVy.noalias() += Xi.transpose() * Vinvyi;
        }
        else
        {
            Eigen::MatrixXd V_inv = V.completeOrthogonalDecomposition().pseudoInverse();
            XVX.noalias() += Xi.transpose() * (V_inv * Xi);
            XVy.noalias() += Xi.transpose() * (V_inv * yi);
        }
        cnt += kt;
    }
    // Final beta solve outside the loop
    Eigen::LDLT<Eigen::MatrixXd> ldlt_XVX(XVX);
    if(ldlt_XVX.info() == Eigen::Success)
    {
        beta = ldlt_XVX.solve(XVy);
    }
    else
    {
        beta = XVX.completeOrthogonalDecomposition().pseudoInverse() * XVy;
    }
}

Eigen::VectorXd R_expand(const Eigen::VectorXd & R,
                         const Eigen::MatrixXi & MAP,
                         int idx, int q)
{
    Eigen::VectorXd R_out = Eigen::VectorXd::Zero(q);
    int cnt = 0;
    for(int i=0;i<q;++i)
    {
        if(MAP(idx,i) == 1)
        {
            R_out(i) = R(cnt);
            cnt++;
        }
    }
    return(R_out);
}


Eigen::MatrixXd RtR(const Eigen::MatrixXd & R, const Eigen::MatrixXi & MAP)//const Eigen::MatrixXd & R, const Eigen::MatrixXi & MAP)
{
    int p = R.cols();

    // 1. Vectorized Square and Zero-Masking
    Eigen::MatrixXd R_sq = (MAP.array() != 0).select(R.array().square().matrix(), 0.0);
    Eigen::MatrixXd MAP_d = MAP.cast<double>();
    
    // 2. BLAS Matrix Multiplication (Instantaneous)
    Eigen::MatrixXd N = MAP_d.transpose() * MAP_d;
    Eigen::MatrixXd SumRsq = R_sq.transpose() * R_sq;
    
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(p, p);

    // 3. Fast Assembly
    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i < p; ++i)
    {
        for(int j=0; j <= i; ++j)
        {
            if(N(i,j) > 0.0) // Protect against divide by zero
            {
                double val = SumRsq(i, j) / N(i,j);
                cov(i, j) = val;
                if(i != j) 
                {
                    cov(j, i) = val;
                }
            }
        }
    }
    return cov;
}
