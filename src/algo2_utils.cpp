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

std::string printdims(Eigen::MatrixXd obj)
{
    return ("("+ std::to_string(obj.rows()) + ", " + std::to_string(obj.cols()) + ")\n");
}

std::string BoolToString(bool b)
{
  return b ? "true" : "false";
}

Eigen::MatrixXd covCalc(const Eigen::MatrixXd & X)
{
    //Eigen::MatrixXd centeredX = X.rowwise() - X.colwise().mean();
    //Eigen::MatrixXd cov = (centeredX.adjoint() * centeredX) / double(X.rows());
    Eigen::MatrixXd cov = ((X.rowwise() - X.colwise().mean()).adjoint() * (X.rowwise() - X.colwise().mean())) / double(X.rows());
    return(cov);
}

void vec2list(const std::vector<Eigen::MatrixXd>& vec, Rcpp::List & out)
{
    for(int i=0;i<vec.size();++i) //(Eigen::MatrixXd i:vec)
    {
        out[i] = vec[i];//out.push_back(i);
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

void estimate_beta(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, const std::vector<Eigen::MatrixXd> & V, Eigen::VectorXd & beta,
    int n, int k, int t)
{
    int q = X.cols();
    Eigen::MatrixXd XVX = Eigen::MatrixXd::Zero(q,q);
    for(int i=0;i<n;++i)
    {
        XVX += X(Eigen::seqN(i*k*t,k*t),Eigen::all).transpose() * V[i].ldlt().solve(X(Eigen::seqN(i*k*t,k*t),Eigen::all));
    }
    Eigen::MatrixXd XVXinvXt = XVX.ldlt().solve(X.transpose());
    beta = Eigen::VectorXd::Zero(q);
    for(int i=0;i<n;++i)
    {
        beta += (V[i].transpose().ldlt().solve(XVXinvXt(Eigen::all,Eigen::seqN(i*k*t,k*t)).transpose())).transpose() * y(Eigen::seqN(i*k*t,k*t));
    }
}


void calc_b(const Eigen::MatrixXd & X, const Eigen::VectorXd & r0, const Eigen::MatrixXd& Z, //Matrix<double, -1,-1,Eigen::RowMajor> & Z,
                       const Eigen::MatrixXd & Lambda_D, const Eigen::MatrixXd & Et, Eigen::VectorXd & b,
                       int n, int k, int t)
{
    int q = Z.cols();
    Eigen::MatrixXd ZEEZ = Eigen::MatrixXd::Zero(q,q);
    Eigen::MatrixXd DZETE = Eigen::MatrixXd::Zero(q,n*k*t);
    Eigen::MatrixXd EtInv = Et.inverse();
    for(int i=0;i<n;i++)
    {
        Eigen::MatrixXd EZ = EtInv * Z(Eigen::seqN(i*k*t,k*t),Eigen::indexing::all);//Et * Z(Eigen::seqN(i*k*t,k*t),Eigen::all);
        ZEEZ += EZ.transpose() * EZ;
        DZETE(Eigen::all,Eigen::seqN(i*k*t,k*t)) = Lambda_D.transpose() * Z(Eigen::seqN(i*k*t,k*t),Eigen::all).transpose() * Et.array().pow(2).matrix().inverse();
    }

    b = Lambda_D.transpose() * (Eigen::MatrixXd::Identity(2*k,2*k) + (Lambda_D.transpose() * ZEEZ * Lambda_D)).colPivHouseholderQr().solve(DZETE) * r0;
}

void estimate_E(const Eigen::MatrixXd & X, const Eigen::VectorXd & r0, const Eigen::MatrixXd& Z, //Matrix<double, -1,-1,Eigen::RowMajor> & Z,
    const Eigen::MatrixXd & Lambda_D, const Eigen::MatrixXd & Et,
    Eigen::VectorXd & Lambda_E, //Eigen::VectorXd & b,
    int n, int k, int t)
{
    Eigen::VectorXd b;
    calc_b(X,r0,Z,Lambda_D,Et,b,n,k,t);
    Eigen::VectorXd r = r0 - Z * b;
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n*t,k);
    for(int i=0;i<k;++i)
    {
        for(int j=0;j<n;++j)
        {
            R(Eigen::seqN(j*t,t),i) = r(Eigen::seqN(j*k*t + i*t,t));
        }
    }
    Lambda_E = covCalc(R).diagonal().array().sqrt();
}

void calc_e(const Eigen::VectorXd & r0, const Eigen::MatrixXd& Z, //const Eigen::MatrixXd& Z, //Matrix<double, -1,-1,Eigen::RowMajor> & Z,
    const Eigen::MatrixXd & Et, const Eigen::MatrixXd & Lambda_D, Eigen::VectorXd & e,
    int n, int k, int t)
{
    int p = Z.cols();
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(p,p);
    for(int i=0;i<n;++i)
    {
        B += Z(Eigen::seqN(i*k*t,k*t),Eigen::all).transpose() * Z(Eigen::seqN(i*k*t,k*t),Eigen::all);
    }
    Eigen::MatrixXd BDBinv = B * Lambda_D * Lambda_D.transpose() * B.transpose();
    Eigen::MatrixXd BD = Lambda_D.colPivHouseholderQr().solve(B.inverse());
    Eigen::MatrixXd BDB = BD.transpose() * BD;
    Eigen::MatrixXd ZEEZ = Eigen::MatrixXd::Zero(p,p);
    Eigen::MatrixXd EZ = Eigen::MatrixXd::Zero(n*k*t,p);
    Eigen::MatrixXd EZ_tmp(k*t,p);
    for(int i=0;i<n;++i)
    {
        EZ_tmp = Et * Z(Eigen::seqN(i*k*t,k*t),Eigen::all);
        EZ(Eigen::seqN(i*k*t,k*t),Eigen::all) = EZ_tmp;
        ZEEZ += EZ_tmp.transpose() * EZ_tmp;
        
    }
    //H = (B.transpose() * Lambda_D.transpose() * Lambda_D * B + ZEEZ).transpose().llt().solve(H.transpose()).transpose();
    Eigen::MatrixXd H = EZ * (BDBinv + ZEEZ).inverse();
    Eigen::VectorXd Zr = Z.transpose() * r0;
    
    for(int i=0;i<n;++i)
    {
        e(Eigen::seqN(i*k*t,k*t)) = Et * (Et * Z(Eigen::seqN(i*k*t,k*t),Eigen::all) - H(Eigen::seqN(i*k*t,k*t),Eigen::all) * ZEEZ) * BDB * Zr;
    }
}


void thresholdRange(const Eigen::MatrixXd & R, Eigen::ArrayXXd& theta, Eigen::MatrixXd& cov, double & lower, double & upper)
{
    int n = R.rows();
    int p = R.cols();
    cov = covCalc(R);

    theta = (R.array().pow(2).matrix().transpose() * R.array().pow(2).matrix() / double(n) - cov.array().pow(2).matrix()).array().sqrt();
    Eigen::MatrixXd delta = (cov.array() / theta).cwiseAbs().matrix();
    delta.diagonal() = Eigen::VectorXd::Zero(p);
    upper = delta.maxCoeff();
    lower = (delta.array() <= 0.f).select(std::numeric_limits<int>::max(), delta).minCoeff();
}

void threshold(const Eigen::MatrixXd& abscov, const Eigen::MatrixXd& signcov, const Eigen::MatrixXd& thetalambda,
                Eigen::MatrixXd& sigma_out)
{
    int p = abscov.rows();
    Eigen::MatrixXd covOffDiag = abscov;
    covOffDiag.diagonal() = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd thetaOffDiag = thetalambda;
    thetaOffDiag.diagonal() = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd sigmaTmp = covOffDiag - thetaOffDiag;
    Eigen::VectorXd sigmaTmpDiag = abscov.diagonal() - thetalambda.diagonal();
    for (int i=0;i<p; ++i)
    {
        if(sigmaTmpDiag(i) <= 0)
        {
            sigmaTmpDiag(i) = 0;
        }
    }
    for (int i=0;i<p; ++i)
    {
        for(int j=0;j<p; ++j)
        {
            if(i == j) continue;
            if(sigmaTmp(i,j) < 0 || sigmaTmpDiag(i) == 0 || sigmaTmpDiag(j) == 0)
            {
                sigmaTmp(i,j) = 0;
            }
        }
    }
    
    sigma_out = sigmaTmp.cwiseProduct(signcov);;
    sigma_out.diagonal() += sigmaTmpDiag;
}

void threshold_D(const Eigen::MatrixXd & R, Eigen::MatrixXd& sigma, Eigen::ArrayXXd & theta, int n_fold=5)
{
    int n = R.rows();
    int p = R.cols();
    int nParam = 100;
    Eigen::MatrixXd cov(p,p);
    double lower = 0.0;
    double upper = 0.0;
    
    thresholdRange(R,theta,cov,lower,upper);
    std::vector<double> params(nParam);
    double jump = (upper - lower)/double(nParam);
    double ctr = lower;
    std::generate(params.begin(), params.end(), [&ctr,&jump]{ return ctr+=jump;});
    auto rng = std::default_random_engine{};
    std::vector<int> part(n);
    std::iota(part.begin(), part.end(), 0);
    std::shuffle(part.begin(),part.end(), rng);
    Eigen::MatrixXd error(n_fold,nParam);
    for (int i=0;i<part.size();++i)
    {
        part[i] = part[i] % n_fold;
    }
    for (int i =0;i<n_fold; ++i)
    {
        std::deque<int> val_idx;
        std::deque<int> not_val_idx;
        find_all(part,i,val_idx,not_val_idx);
        Eigen::MatrixXd covTest = covCalc(R(val_idx,Eigen::all));
        double lower = 0.0;
        double upper = 0.0;
        Eigen::MatrixXd absCovTrain(p,p);
        Eigen::ArrayXXd thetaTrain(p,p);
        thresholdRange(R(not_val_idx,Eigen::all),thetaTrain,absCovTrain,lower,upper);
        Eigen::ArrayXXd signCovTrain = absCovTrain.array() / absCovTrain.cwiseAbs().array();
        absCovTrain = absCovTrain.cwiseAbs();
        for(int j=0;j<nParam;++j)
        {

            //Eigen::MatrixXd thetalambda = params[j] * thetaTrain;
            Eigen::MatrixXd sigmaTrain(p,p);
            threshold(absCovTrain,signCovTrain,params[j] * thetaTrain,sigmaTrain);
            error(i,j) = (sigmaTrain - covTest).norm();
        }
    }
    Eigen::Index minIndex;
    error.colwise().sum().minCoeff(&minIndex);
    Eigen::ArrayXXd absCov = cov.cwiseAbs();
    Eigen::ArrayXXd signCov = cov.array() / absCov;

    threshold(absCov,signCov,params[minIndex] * theta,sigma);

}


void estimate_D(const Eigen::MatrixXd & X, const Eigen::VectorXd & r0, const Eigen::MatrixXd& Z, //Matrix<double, -1,-1,Eigen::RowMajor> & Z,
    const Eigen::MatrixXd & Et,
    Eigen::MatrixXd & Lambda_D, Eigen::MatrixXd & D,
    int n, int k, int t, bool soft=1)
{
    int p = Z.cols();
    Eigen::VectorXd e = Eigen::VectorXd::Zero(n*k*t);
    calc_e(r0,Z,Et,Lambda_D,e,n,k,t);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n,p);
    for(int i=0;i<n;++i)
    {
        R(i,Eigen::all) = (Z(Eigen::seqN(i*k*t,k*t),Eigen::all).transpose() * Z(Eigen::seqN(i*k*t,k*t),Eigen::all)).colPivHouseholderQr().solve(Z(Eigen::seqN(i*k*t,k*t),Eigen::all).transpose()) * (r0(Eigen::seqN(i*k*t,k*t))-e(Eigen::seqN(i*k*t,k*t)));
    }


    
    Eigen::ArrayXXd theta(2*k,2*k);
    threshold_D(R,D,theta);
    Lambda_D = (D + Eigen::MatrixXd::Identity(p,p)).llt().matrixL();

}

double calc_sigma2(const Eigen::MatrixXd& Z, const Eigen::MatrixXd& D, const Eigen::MatrixXd& E, const Eigen::VectorXd& r0, int n, int k, int t, int p, bool REML=false)
{
    double sigma2 = 0.0;
    Eigen::MatrixXd Lambda_V;
    for(int i=0; i<n;++i)
    {
      Lambda_V = (Z(Eigen::seqN(i*k*t,k*t),Eigen::all) * D * Z(Eigen::seqN(i*k*t,k*t),Eigen::all).transpose()  + E).llt().matrixL();
      sigma2 += (Lambda_V.colPivHouseholderQr().solve(r0(Eigen::seqN(i*k*t,k*t)))).squaredNorm();
      
    }
    if(REML)
    {
        sigma2 = sigma2 / (n*k*t - p);
    } else {
        sigma2 = sigma2 / (n*k*t);
    }
    return(sigma2);
}

void calc_ZDZ_plus_E_list(const Eigen::MatrixXd& Z, //Matrix<double, -1,-1,Eigen::RowMajor> & Z, 
                          const Eigen::MatrixXd & D, const Eigen::MatrixXd & E,
                     std::vector<Eigen::MatrixXd> & out,
                     int n, int k, int t)
{
    for(int i=0; i<n;++i)
    {
        out[i] = (Z(Eigen::seqN(i*k*t,k*t),Eigen::all) * D * Z(Eigen::seqN(i*k*t,k*t),Eigen::all).transpose() + E);
    }
}

void initial_estimates(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, const Eigen::MatrixXd & Z, 
    std::vector<Eigen::MatrixXd> & Sigma_list, Eigen::MatrixXd & Lambda_D, Eigen::MatrixXd & D, 
    Eigen::VectorXd & Lambda_E, Eigen::VectorXd & beta, Eigen::VectorXd & r,
    int n, int k, int t)
{
    int p = X.cols();
    int q = Z.cols();
    beta = (X.transpose() * X).ldlt().solve(X.transpose()) * y;
    r = y - X * beta;
    Eigen::MatrixXd R(n,k*t);
    for(int i=0;i<n;++i)
    {
        R(i,Eigen::seqN(0,k*t)) = r(Eigen::seqN(i*k*t,k*t));
    }
    Eigen::MatrixXd cov_int = covCalc(R);
    for(int i=0;i<k;++i)
    {
        Lambda_E(i) = (cov_int.diagonal().array())(Eigen::seqN(i*t,t)).sqrt().mean() / 2;
    }

    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(q,q); 
    Eigen::MatrixXd ZCZ = Eigen::MatrixXd::Zero(q,q);
    for(int i=0;i<n;++i)
    {
        B += Z(Eigen::seqN(i*k*t,k*t),Eigen::all).transpose() * Z(Eigen::seqN(i*k*t,k*t),Eigen::all);
        ZCZ += Z(Eigen::seqN(i*k*t,k*t),Eigen::all).transpose() * cov_int * Z(Eigen::seqN(i*k*t,k*t),Eigen::all);
    }
    D = B.colPivHouseholderQr().solve(ZCZ) * B.inverse();
    Lambda_D = (D + Eigen::MatrixXd::Identity(q,q)).llt().matrixL();
    Eigen::MatrixXd E = Eigen::MatrixXd::Zero(k*t,k*t);
    for(int i=0;i<t;++i)
    {
        E(Eigen::seqN(i,k,t),Eigen::seqN(i,k,t)).diagonal() = Lambda_E.array().pow(2);
    }
    calc_ZDZ_plus_E_list(Z,D,E,Sigma_list,n,k,t);
}

// [[Rcpp::export]]
Rcpp::List estimate_DEbeta(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, const Eigen::MatrixXd & Z, int n, int k, int t, int max_itr=250, std::string covtype="", int idx=0, bool REML=false, double eigen_threshold=0.001)
{
    int p = X.cols();
    std::vector<Eigen::MatrixXd> Sigma_list(n);
    Eigen::MatrixXd Lambda_D(2*k,2*k);
    Eigen::MatrixXd D(2*k,2*k);
    Eigen::VectorXd Lambda_E(k);
    Eigen::VectorXd beta(p);
    Eigen::VectorXd r0(n*k*t);
    initial_estimates(X,y,Z,Sigma_list,Lambda_D,D,Lambda_E,beta,r0,n,k,t);
    
    double err = 10.0;
    double prev_err = 10.0;
    int n_itr = 0;
    std::vector<double> all_err(max_itr);
    Eigen::MatrixXd Lambda_Et = Eigen::MatrixXd::Zero(k*t,k*t);
    for(int i=0;i<t;++i)
    {
        Lambda_Et(Eigen::seqN(i,k,t),Eigen::seqN(i,k,t)).diagonal() = Lambda_E;
    }
    while (((err > (5*pow(10,-4))) || (prev_err > (5*pow(10,-4)))) && (n_itr < max_itr))
    {
        prev_err = err;
        Eigen::MatrixXd D_prev = Lambda_D;
        Eigen::VectorXd E_prev = Lambda_E;
        Eigen::VectorXd beta_prev = beta;

        estimate_beta(X,y,Sigma_list,beta,n,k,t);
        r0 = y - X * beta; 
        estimate_D(X,r0,Z,Lambda_Et,Lambda_D,D,n,k,t);
        estimate_E(X,r0,Z,Lambda_D,Lambda_Et,Lambda_E,n,k,t);

        for(int i=0;i<t;++i)
        {
            Lambda_Et(Eigen::seqN(i,k,t),Eigen::seqN(i,k,t)).diagonal() = Lambda_E;
        }
        calc_ZDZ_plus_E_list(Z,D,Lambda_Et.array().pow(2), Sigma_list,n, k, t);

        err = (Lambda_D - D_prev).squaredNorm() / (k*(2*k+1)) + (Lambda_E - E_prev).squaredNorm() / k + pow((beta - beta_prev).sum(),2) / p;
        err = err / 3;
        all_err[n_itr] = err;
        n_itr++;
    }
    Eigen::VectorXd E0 = Lambda_E.array().pow(2);
    Eigen::MatrixXd Et = Eigen::MatrixXd::Zero(k*t,k*t);
    for(int i=0;i<t;++i)
    {
        Et(Eigen::seqN(i,k,t),Eigen::seqN(i,k,t)).diagonal() = E0;
    }
    double sigma2 = calc_sigma2(Z,D,Et,r0,n,k,t,p,REML);
    D = D * sigma2;
    Et = Et * sigma2;
    //calc_ZDZ_plus_E_list(Z,D,Et,Sigma_list,n,k,t);

    bool converged = false;
    if(n_itr < max_itr) converged = true;

    Eigen::MatrixXd E = Eigen::MatrixXd::Zero(k,k);
    E.diagonal() = E0 * sigma2;
    //Rcpp::List Sigma(n); 
    //vec2list(Sigma_list, Sigma);
    return(Rcpp::List::create(//Rcpp::Named("Sigma")=Sigma,
           Rcpp::Named("E") = E,Rcpp::Named("D") = D, 
           Rcpp::Named("Lambda_E") = Lambda_E, 
           Rcpp::Named("beta") = beta, 
           Rcpp::Named("n_iter") = n_itr,
           Rcpp::Named("all_err") = all_err,
           Rcpp::Named("converged") = converged, 
           Rcpp::Named("sigma") = sigma2));
}