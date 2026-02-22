#define STRICT_R_HEADERS
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
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(p,p);

    for(int i=0; i < p; ++i)
    {
        for(int j=0; j <= i; ++j)
        {
            double xmean = 0.0;
            double ymean = 0.0;
            int n_obs = 0;

            for(int l=0; l < n; ++l)
            {
                if(MAP(l,i) != 0 && MAP(l,j) != 0)
                {
                    n_obs++;
                    xmean += X(l,i);
                    ymean += X(l,j);
                }
            }
            if(n_obs == 0)
            {
                continue;
            }
            xmean = xmean / (n_obs);
            ymean = ymean / (n_obs);
            for(int l=0; l < n; ++l)
            {
                if(MAP(l,i) != 0 && MAP(l,j) != 0)
                {
                    cov(i,j) += ((X(l,i) - xmean) * (X(l,j) - ymean) / (n_obs));
                    if(i != j)
                    {
                        cov(j,i) += ((X(l,i) - xmean) * (X(l,j) - ymean) / (n_obs));
                    }
                } 
            }

            if(print)
            {    
                Rcpp::Rcout << "n_obs=" << n_obs << "\n";
                Rcpp::Rcout << "xmean=" << xmean << "\n";
                Rcpp::Rcout << "ymean=" << ymean << "\n";
            }
        }
    }
    return(cov);
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
                    const std::vector<Eigen::MatrixXd> & Z,
                    const Eigen::MatrixXd & D,
                    const Eigen::VectorXd & E,
                    const Eigen::VectorXi kt_vec, const Eigen::MatrixXi & MAP,
                    Eigen::VectorXd & beta,
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
        Eigen::MatrixXd V = (Z[i] * D * Z[i].transpose() + Et_assemble(E, MAP, i, k, t, kt));
        Eigen::FullPivLU<Eigen::MatrixXd> lu(V);
        v_inv[i] = lu.isInvertible();
        if(v_inv[i])
        {
            XVX += X(Eigen::seqN(cnt,kt),Eigen::all).transpose() * V.colPivHouseholderQr().solve(X(Eigen::seqN(cnt,kt),Eigen::all));
        }
        else
        {
            //Rcpp::Rcout << "Singular matrix - using pseudoinv" << "\n";
            //Rcpp::Rcout << "idx = " << idx << " sjt= " << i << "\n";
            //Rcpp::Rcout << "n0 = " << n << "; k0= " << k << "; t0=" << t << "\n";
            V_inv[i] = V.completeOrthogonalDecomposition().pseudoInverse();
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
            Eigen::MatrixXd V = (Z[i] * D * Z[i].transpose() + Et_assemble(E, MAP, i, k, t, kt));
            beta += XVXinvXt(Eigen::all,Eigen::seqN(cnt,kt)) * V.colPivHouseholderQr().solve(y(Eigen::seqN(cnt,kt)));
        }
        else
        {
            beta += XVXinvXt(Eigen::all,Eigen::seqN(cnt,kt)) * V_inv[i] * y(Eigen::seqN(cnt,kt));
        }
        cnt += kt;
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
    int n = R.rows();
    int p = R.cols();
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(p,p);
    //Eigen::ArrayXXd R_arr = R.array().pow(2).matrix();

    for(int i=0; i < p; ++i)
    {
        for(int j=0; j <= i; ++j)
        {
            double xmean = 0.0;
            double ymean = 0.0;
            int n_obs = 0;

            for(int l=0; l < n; ++l)
            {
                if(MAP(l,i) != 0 && MAP(l,j) != 0)
                {
                    n_obs++;
                    xmean += R(l,i);
                    ymean += R(l,j);
                }
            }
            if(n_obs == 0)
            {
                continue;
            }
            xmean = xmean / n_obs;
            ymean = ymean / n_obs;
            for(int l=0; l < n; ++l)
            {
                if(MAP(l,i) != 0 && MAP(l,j) != 0)
                {
                     /*cov(i,j) += (pow(R(l,i) - xmean,2) * pow(R(l,j) - ymean,2)) / double(n_obs);
                    if(i != j)
                    {
                        cov(j,i) += (pow(R(l,i) - xmean,2) * pow(R(l,j) - ymean,2)) / double(n_obs);
                    }*/
                    cov(i,j) += (pow(R(l,i),2) * pow(R(l,j),2)) / double(n_obs);
                    if(i != j)
                    {
                        cov(j,i) += (pow(R(l,i),2) * pow(R(l,j),2)) / double(n_obs);
                    }
                } 
            }
        }
    }
    return(cov);
}
