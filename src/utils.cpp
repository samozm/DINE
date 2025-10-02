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

std::string printdims(const Eigen::MatrixXd & obj)
{
    return ("("+ std::to_string(obj.rows()) + ", " + std::to_string(obj.cols()) + ")\n");
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
    bool out = false;
    for(int i: b)
    {
        if(a == i)
        {
            return(1);
        }
    }
    return(0);
}

void build_V_list_from_master(std::vector<Eigen::MatrixXd> & V, const Eigen::MatrixXd & master, const Eigen::MatrixXi & MAP, int n, int k, int t)
{
  for(int i=0; i<n; ++i)
  {
    int kt0 = MAP.rowwise().sum()(i);
    V[i] = Eigen::MatrixXd::Zero(kt0,kt0);
    int t00 = kt0 / k;
    kt0 = 0;
    for(int j=0;j<k;++j)
    {
      int t0 = 0;
      for(int l=0;l<t;++l)
      {
        if(MAP(i,kt0) == 1)
        {    
          int kt1 = 0;
          for(int m=0;m<k;++m)
          {
            int t1 = 0;
            for(int o=0;o<t;++o)
            {
              if(MAP(i,kt1) == 1)
              {
                // (j*t00 + t0,m*t00 + t1) = cov_int(kt0,kt1);
                V[i](j*t00 + t0,m*t00 + t1) = master(kt0, kt1);
                t1++;
              }
              kt1++;
            }
          }
          t0++;
        }
        kt0++;
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
Eigen::MatrixXd covCalc(const Eigen::MatrixXd & X, const Eigen::MatrixXi & MAP, bool print=false)
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
    for(int i=0; i<n;++i)
    {
        int kt = Z[i].rows();
        out[i] = (Z[i] * D * Z[i].transpose() + Et_assemble(E, MAP, i, k, t, kt));
    }
}

void estimate_beta(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, const Eigen::VectorXi kt_vec,
                   const std::vector<Eigen::MatrixXd> & V, Eigen::VectorXd & beta,
                   int n, int k, int t)
{
    int q = X.cols();
    Eigen::MatrixXd XVX = Eigen::MatrixXd::Zero(q,q);
    int cnt = 0;
    for(int i=0;i<n;++i)
    {
        int kt = kt_vec(i);
        XVX += X(Eigen::seqN(cnt,kt),Eigen::all).transpose() * V[i].ldlt().solve(X(Eigen::seqN(cnt,kt),Eigen::all));
        cnt += kt;
    }
    Eigen::MatrixXd XVXinvXt = XVX.ldlt().solve(X.transpose());
    beta = Eigen::VectorXd::Zero(q);
    cnt = 0;
    for(int i=0;i<n;++i)
    {
        int kt = kt_vec(i);
        beta += (V[i].transpose().ldlt().solve(XVXinvXt(Eigen::all,Eigen::seqN(cnt,kt)).transpose())).transpose() * y(Eigen::seqN(cnt,kt));
        cnt += kt;
    }
}

