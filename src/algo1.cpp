#define STRICT_R_HEADERS
#include "utils.h"


// [[Rcpp::export]]
Rcpp::List estimate_V(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, 
    std::vector<Eigen::MatrixXd> & V, const Eigen::VectorXd & beta,
    const Eigen::MatrixXi & MAP,
    int n, int k, int t)
{
    Eigen::VectorXd r0 = y - X * beta;
    V = std::vector<Eigen::MatrixXd>(n);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n,k*t);
    int current = 0;
    for(int i = 0; i<n; ++i)
    {
        int kt0 = MAP.rowwise().sum()(i);
        std::deque<int> idxs;
        std::deque<int> waste;
        find_all(MAP(i,Eigen::all),1,idxs,waste);
        R(i,idxs) = r0(Eigen::seqN(current,kt0));
        current += kt0;
    }
    Eigen::MatrixXd master_V = covCalc(R,MAP);
    build_V_list_from_master(V,master_V,MAP,n,k,t);
    return(Rcpp::List::create(Rcpp::Named("V")=V,Rcpp::Named("R")=R,Rcpp::Named("MasterV")=master_V));
}


// [[Rcpp::export]]
Rcpp::List estimate_D(const Eigen::VectorXd & r0, const std::vector<Eigen::MatrixXd> & Z,
                Eigen::MatrixXd & D, const Eigen::MatrixXd & E, 
                const std::vector<Eigen::MatrixXd> & V, const Eigen::MatrixXi & MAP, 
                int n, int k, int t)
{
    //Eigen::MatrixXd D_minus_E = D - E;
    D = Eigen::MatrixXd::Zero(2*k,2*k);
    Eigen::MatrixXd ZTZ = Eigen::MatrixXd::Zero(2*k,2*k);
    for(int i=0; i<n; ++i)
    {
        ZTZ += Z[i].transpose() * Z[i];
    }
    for(int i=0; i<n; ++i)
    {
        if(i == 0)
        {
            Rcpp::Rcout << "Vi" << printdims(V[i]) << "\n";
            Rcpp::Rcout << V[i](Eigen::seqN(0,5), Eigen::seqN(0,5)) << "\n";

            Rcpp::Rcout << "E" << printdims(E) << "\n";
            Rcpp::Rcout << E(Eigen::seqN(0,5), Eigen::seqN(0,5)) << "\n";

            Rcpp::Rcout << "Zi" << printdims(Z[i]) << "\n";
            Rcpp::Rcout << Z[i](Eigen::seqN(0,5), Eigen::seqN(0,5)) << "\n";
        }
        Eigen::MatrixXd P = ZTZ.colPivHouseholderQr().solve(Z[i].transpose());
        if(i == 0)
        {
            Rcpp::Rcout << "P(0)" << printdims(P) << "\n";
            Rcpp::Rcout << P(Eigen::seqN(0,5), Eigen::seqN(0,5)) << "\n";
        }
        D += P * (V[i] - E) * P.transpose();
    }
    return(Rcpp::List::create(Rcpp::Named("D")=D));
}

Rcpp::List estimate_E(const Eigen::VectorXd & r0, const std::vector<Eigen::MatrixXd> & Z,
                const Eigen::MatrixXd & D, Eigen::MatrixXd & E, 
                const std::vector<Eigen::MatrixXd> & V, const Eigen::MatrixXi & MAP, 
                int n, int k, int t)
{
    // We only need the diagonals of the ZDZ matrix
    std::vector<Eigen::MatrixXd> ZDZ(n); //Z * D * Z.transpose();
    for(int i=0; i<n; ++i)
    {   
        // We only need the diagonals of the ZDZ matrix
        ZDZ[i] = Z[i] * D * Z[i].transpose();
    }
    Eigen::MatrixXd E_tmp = Eigen::MatrixXd::Zero(k,t);
    // because not every subject has the same timepoints, keep track of the current timepoint separately for each subject
    Eigen::VectorXd ct = Eigen::VectorXd::Zero(n);
    for(int i=0;i<k;++i)
    {
        for(int j=0;j<t;++j)
        {
            int bot = 0;
            for(int l=0;l<n;++l)
            {
                if(MAP(l,i*t + j) == 1)
                {
                    Eigen::MatrixXd Vl = V[l];
                    Eigen::MatrixXd ZDZl = ZDZ[l];
                    bot++;
                    double A = E_tmp(i,j);
                    // Just the most recent time for subject l
                    double B = Vl.coeff(ct(l),ct(l));
                    double C = ZDZl.coeff(ct(l),ct(l));
                    E_tmp(i,j) = A + B - C; //((double) E_tmp(i,j)) + ((double) Vl(i*k+ct(l),i*k+ct(l)) - ZDZ(l*n + i*k + ct(l),l*n + i*k + ct(l));)
                    ct(l)++;
                    if(i == 0 && j == 0)
                    {
                        Rcpp::Rcout << B-C << ", ";
                    }
                }
            }
            if(i == 0 && j == 0)
            {
                Rcpp::Rcout << "/ " << bot << "\n";
            }
            E_tmp(i,j) /= bot;
            if(E_tmp(i,j) <= 0)
            {
                E_tmp(i,j) = double(1)/(n*k);
            }
        }
    }
    Rcpp::Rcout << '\n';
    E = Eigen::MatrixXd::Zero(k,k);
    E.diagonal() = E_tmp.rowwise().mean();
    for(int i=0;i<k;++i)
    {
        E(i,i) = E(i,i) > 0 ? E(i,i) : double(1) /(n*k);
    }
    Rcpp::List ZDZ_list(n);
    vec2list(ZDZ,ZDZ_list);
    return(Rcpp::List::create(Rcpp::Named("E")=E,Rcpp::Named("E_tmp")=E_tmp,Rcpp::Named("ZDZ")=ZDZ_list));
}

// [[Rcpp::export]]
Rcpp::List estimate_E(const Eigen::VectorXd & r0, const Rcpp::List & Z,
                      const Eigen::MatrixXd & D, Eigen::MatrixXd & E, 
                      const std::vector<Eigen::MatrixXd> & V, 
                      const Eigen::MatrixXi & MAP, 
                      int n, int k, int t)
  
{
    std::vector<Eigen::MatrixXd> Z_vec(n);
    list2vec(Z_vec,Z);
    return(estimate_E(r0,Z_vec,D,E,V,MAP,n,k,t));
}

// [[Rcpp::export]]
void thresholdRange(const Eigen::MatrixXd & R, Eigen::ArrayXXd& theta, Eigen::MatrixXd& cov, const Eigen::MatrixXi & MAP, double & lower, double & upper)
{
    int n = R.rows();
    int p = R.cols();
    cov = covCalc(R,MAP);

    theta = (R.array().pow(2).matrix().transpose() * R.array().pow(2).matrix() / double(n) - cov.array().pow(2).matrix()).array().sqrt();
    Eigen::MatrixXd delta = (cov.array() / theta).cwiseAbs().matrix();
    delta.diagonal() = Eigen::VectorXd::Zero(p);
    upper = delta.maxCoeff();
    lower = (delta.array() <= 0.f).select(std::numeric_limits<int>::max(), delta).minCoeff();
    /*return(Rcpp::List::create(Rcpp::Named("theta")=theta,
                              Rcpp::Named("upper")=upper,
                              Rcpp::Named("lower")=lower,
                              Rcpp::Named("delta")=delta));*/
}


// [[Rcpp::export]]
void threshold(const Eigen::MatrixXd& abscov, const Eigen::MatrixXd& signcov, const Eigen::MatrixXd& thetalambda, Eigen::MatrixXd& sigma_out)
{
    int p = abscov.rows();
    Eigen::MatrixXd covOffDiag = abscov;
    covOffDiag.diagonal() = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd thetaOffDiag = thetalambda;
    thetaOffDiag.diagonal() = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd sigmaTmp = covOffDiag - thetaOffDiag;

    for (int i=0;i<p; ++i)
    {
      for(int j=0;j<i; ++j)
      {
        //if(i == j) continue;
        if(sigmaTmp(i,j) < 0)
        {
          sigmaTmp(i,j) = 0;
          sigmaTmp(j,i) = 0;
        }
      }
    }
    
    sigma_out = sigmaTmp.cwiseProduct(signcov);;
    sigma_out.diagonal() += abscov.diagonal();
    /*return(Rcpp::List::create(Rcpp::Named("sigma")=sigma_out,
                              Rcpp::Named("sigmaTmp")=sigmaTmp,
                              Rcpp::Named("sigmaTmpDiag")=sigmaTmpDiag,
                              Rcpp::Named("diag_zeros")=diag_zeros));*/
}

// [[Rcpp::export]]
Rcpp::List threshold_V(const Eigen::MatrixXd & R, Eigen::MatrixXd& sigma, Eigen::ArrayXXd & theta, const Eigen::MatrixXi & MAP, int n_fold=5)
{
    int n = R.rows();
    int p = R.cols();
    int nParam = 100;
    Eigen::MatrixXd cov(p,p);
    double lower = 0.0;
    double upper = 0.0;

    thresholdRange(R,theta,cov,MAP,lower,upper);
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
        Rcpp::Rcout << "fold = " << i << " " << val_idx.size() << " values" << "\n";
        Eigen::MatrixXd covTest = covCalc(R(val_idx,Eigen::all),MAP(val_idx,Eigen::all));
        double lower = 0.0;
        double upper = 0.0;
        Eigen::MatrixXd absCovTrain(p,p);
        Eigen::ArrayXXd thetaTrain(p,p);
    thresholdRange(R(not_val_idx,Eigen::all),thetaTrain,absCovTrain,MAP(not_val_idx,Eigen::all),lower,upper);
        Eigen::ArrayXXd signCovTrain = absCovTrain.cwiseSign(); //absCovTrain.array() / absCovTrain.cwiseAbs().array();
        absCovTrain = absCovTrain.cwiseAbs();
        for(int j=0;j<nParam;++j)
        {

            Eigen::MatrixXd thetalambda = params[j] * thetaTrain;
            Eigen::MatrixXd sigmaTrain(p,p);
            threshold(absCovTrain,signCovTrain,params[j] * thetaTrain,sigmaTrain);
            error(i,j) = (sigmaTrain - covTest).norm();
        }
    }
    Eigen::Index minIndex;
    error.colwise().sum().minCoeff(&minIndex);
    //Rcpp::Rcout << "err" << "\n";
    //Rcpp::Rcout << error.colwise().sum() << "\n";
    Eigen::ArrayXXd absCov = cov.cwiseAbs();
    Eigen::ArrayXXd signCov = cov.array() / absCov;

    threshold(absCov,signCov,params[minIndex] * theta,sigma);

    //Rcpp::Rcout << "theta" << printdims(theta) << "\n";
    //Rcpp::Rcout << theta(Eigen::seqN(0,5), Eigen::seqN(0,5)) << "\n";
    return(Rcpp::List::create(Rcpp::Named("V") = sigma));
}


// [[Rcpp::export]]
Rcpp::List calc_ZDZ_wrapper(const std::vector<Eigen::MatrixXd>& Z, //Matrix<double, -1,-1,Eigen::RowMajor> & Z, 
                            const Eigen::MatrixXd & D, const Eigen::VectorXd & E,
                            const Eigen::MatrixXi & MAP,
                            int n, int k, int t)
{
    std::vector<Eigen::MatrixXd> out(n);
    calc_ZDZ_plus_E_list(Z,D,E,out,MAP,n,k,t);
    return(Rcpp::List::create(Rcpp::Named("Sigma")=out));
}

// [[Rcpp::export]]
Rcpp::List threshold_D(Eigen::MatrixXd & D, double nonzero_pct)
{
    int p = D.rows();
    
    int cutoff = std::round(p*p*nonzero_pct) - 1;
    Eigen::VectorXd DoffVecSorted = D.cwiseAbs().reshaped();
    std::sort(DoffVecSorted.begin(), DoffVecSorted.end(), [](double const& t1, double const& t2){ return t1 < t2; } );
    double threshold = cutoff == 0 ? 0 : DoffVecSorted[cutoff];
    Eigen::MatrixXd thresholdMat = Eigen::MatrixXd::Constant(p,p,1.0) * threshold;
    Eigen::MatrixXd D_tmp = D.cwiseAbs() - thresholdMat;

    Rcpp::Rcout << "cutoff = " << cutoff << "\n";
    Rcpp::Rcout << "threshold = " << threshold << "\n";
    Rcpp::Rcout << "threshold = " << thresholdMat << "\n";
    Rcpp::Rcout << "DoffVecSorted = " << DoffVecSorted(Eigen::seqN(0,5)) << "\n";

    for (int i=0;i<p; ++i)
    {
        for(int j=0;j<p; ++j)
        {
            if(D_tmp(i,j) < 0)
            {
                D_tmp(i,j) = 0;
            }
        }
    }
    D = D_tmp.cwiseProduct(D.cwiseSign());
    return(Rcpp::List::create(Rcpp::Named("D")=D));
}



// [[Rcpp::export]]
Rcpp::List initial_estimates(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, 
    const Eigen::MatrixXi & MAP,
    int n, int k, int t)//, vector<int> t)
{
  
    std::vector<Eigen::MatrixXd> Sigma_list(n);
  
    int p = X.cols();
    Eigen::VectorXd beta = (X.transpose() * X).llt().solve(X.transpose()) * y;
    Eigen::VectorXd r_out = y - X * beta;
    Rcpp::Rcout << printdims(r_out) << "\n";
    Rcpp::Rcout << r_out(Eigen::seqN(0,5)) << "\n";
    Eigen::MatrixXd R(n,k*t);
    int current = 0;
    for(int i = 0; i<n; ++i)
    {
        int kt0 = MAP.rowwise().sum()(i);
        std::deque<int> idxs;
        std::deque<int> waste;
        find_all(MAP(i,Eigen::all),1,idxs,waste);
        R(i,idxs) = r_out(Eigen::seqN(current,kt0));
        current += kt0;
    }
    Eigen::MatrixXd cov_int = covCalc(R,MAP);
    for(int i=0; i<n; ++i)
    {
        int kt0 = MAP.rowwise().sum()(i);
        Sigma_list[i] = Eigen::MatrixXd::Zero(kt0,kt0);
        int t00 = kt0/k;
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
                                Sigma_list[i](j*t00 + t0,m*t00 + t1) = cov_int(kt0, kt1);
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

    return(Rcpp::List::create(Rcpp::Named("Sigma_list")=Sigma_list, 
            Rcpp::Named("beta")=beta,Rcpp::Named("r")=r_out,
            Rcpp::Named("cov")=cov_int));
}

void initial_estimates(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, 
                       Eigen::VectorXd & r_out, Eigen::VectorXd & beta, 
                       std::vector<Eigen::MatrixXd> & Sigma_list, const Eigen::MatrixXi & MAP,
                       int n, int k, int t)
{
    auto list_out = initial_estimates(X,y,MAP,n,k,t);
    list2vec(Sigma_list,list_out[0]);
    r_out = list_out[2];
    beta = list_out[1];
}

int estimate_DE(Eigen::VectorXd& r0, const std::vector<Eigen::MatrixXd> & Z, std::vector<Eigen::MatrixXd> & Sigma_list, const Eigen::MatrixXi & MAP, int n, int k, int t, double V_nonzeros_pct, int max_itr, Eigen::MatrixXd & E, Eigen::MatrixXd & D)
{
    D = Eigen::MatrixXd::Identity(2*k,2*k);
    Eigen::MatrixXd E_tmp = Eigen::MatrixXd::Identity(k*t,k*t); 
    double err = 10.;
    double prev_err = 10.;
    int n_itr = 0;

    while (((err > (5*pow(10,-4))) || (prev_err > (5*pow(10,-4)))) && (n_itr < max_itr))
    {
        Eigen::MatrixXd D_prev = D;
        Eigen::MatrixXd E_prev = E_tmp;

        estimate_D(r0,Z,D,E_tmp,Sigma_list,MAP,n,k,t);
        estimate_E(r0,Z,D,E_tmp,Sigma_list,MAP,n,k,t);

        prev_err = err;
        err = ((D - D_prev).squaredNorm() / (4*k^2) + (E_tmp - E_prev).squaredNorm() / k) / 2;
        n_itr++;
    }
    int cnt = 0; 
    for(int i=0;i<k;++i)
    {
        int t00 = MAP.rowwise().sum()(i)/k;
        E(i,i) = E_tmp(Eigen::seqN(cnt,t00),Eigen::seqN(cnt,t00)).diagonal().mean();
        cnt += t00;
    }
    threshold_D(D,V_nonzeros_pct);
    return(BoolToInt(n_itr < max_itr-1));
}

int sigma_norm_diff(const std::vector<Eigen::MatrixXd> & A,const std::vector<Eigen::MatrixXd> & B, int n)
{
  int out = 0;
  for(int i=0; i<n; ++i)
  {
    out += (A[i] - B[i]).squaredNorm();
  }
  return(out);
  
}


int estimate_betaV(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, Eigen::VectorXd & beta, std::vector<Eigen::MatrixXd> & V, const Eigen::MatrixXi & MAP, const Eigen::VectorXi kt_vec, int n, int k, int t, int max_itr)
{
    int p = X.cols();
    beta = (X.transpose() * X).colPivHouseholderQr().solve(X.transpose()) * y;
    double err = 10.;
    double prev_err = 10.;
    int n_itr = 0;

    while (((err > (5*pow(10,-4))) || (prev_err > (5*pow(10,-4)))) && (n_itr < max_itr))
    {
        Eigen::VectorXd beta_prev = beta;
        std::vector<Eigen::MatrixXd> V_prev = V;

        estimate_beta(X,y,kt_vec,V,beta,n,k,t);
        estimate_V(X,y,V,beta,MAP,n,k,t);

        prev_err = err;
        err = ((beta - beta_prev).squaredNorm() / p + (sigma_norm_diff(V,V_prev,n)/ (n*pow(t*k,2)))) / 2;
        n_itr++;
    }

    Eigen::MatrixXd master_V(t*k,t*k);
    Eigen::ArrayXXd theta(2*k,2*k);
    Eigen::MatrixXd R(n,k*t);
    Eigen::VectorXd r0 = y - X * beta;
    int current = 0;
    
    for(int i = 0; i<n; ++i)
    {
      int kt0 = MAP.rowwise().sum()(i);
      std::deque<int> idxs;
      std::deque<int> waste;
      find_all(MAP(i,Eigen::all),1,idxs,waste);
      R(i,idxs) = r0(Eigen::seqN(current,kt0));
      current += kt0;
    }
    
    threshold_V(R,master_V,theta,MAP);
    build_V_list_from_master(V,master_V,MAP,n,k,t);
    return(BoolToInt(n_itr < max_itr-1));
}


// [[Rcpp::export]]
Rcpp::List estimate_all(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, const Rcpp::List & Z_in, int n, int k, int t, int max_itr=250, std::string covtype="", int idx=0, bool REML=false, double eigen_threshold=0.001)
{
    int p = X.cols();
    std::vector<Eigen::MatrixXd> Sigma_vec(n);
    std::vector<Eigen::MatrixXd> Z(n);
    list2vec(Z,Z_in);
    Rcpp::List Sigma_list(n);
    Eigen::MatrixXd D(2*k,2*k);
    Eigen::MatrixXd E(k,k);
    Eigen::VectorXd beta(p);
    //X, y, Z, r_out, beta, Sigma_list, MAP,
    std::deque<double> all_times; //= Z(Eigen::all,Eigen::seqN(1,2*k,2)).reshaped();
    int nkt = 0;
    for(int i=0; i<n; ++i)
    {
        nkt += Z[i].rows();
        Eigen::VectorXd n_times = Z[i](Eigen::all,Eigen::seqN(1,2*k,2)).reshaped();
        for(int j=0;j<n_times.size();++j)
        {
          all_times.push_back(n_times(j)); 
        }
    }
    Eigen::VectorXd r0(nkt);
    std::set<double> time_set{all_times.begin(), all_times.end()};
    std::vector<double> times(time_set.begin(), time_set.end());
    Eigen::MatrixXi MAP = Eigen::MatrixXi::Zero(n,k*t);
    for(int i = 0; i < n; ++i)
    {
        for(int j=0; j < t; ++j)
        {
            for(int l=0; l<k; ++l)
            {
                MAP(i,j*t + l) = a_in_b(times[j],Z[i](Eigen::all,2*l + 1));
            }
        }
    }
    initial_estimates(X,y,r0,beta,Sigma_vec,MAP,n,k,t);
    //Sigma_list = init_est[0];
    //beta = init_est[1];
    //r0 = init_est[2];
    
    int converged = estimate_betaV(X,y,beta,Sigma_vec,MAP,MAP.rowwise().sum(),n,k,t,max_itr);
    double V_nonzeros_pct = 0;
    int denom = 0;
    for(int i=0; i<n; ++i)
    {
        int kt0 = Z[i].rows();
        for(int j=0; j<kt0; ++j)
        {
            for(int l=0;l<j;++l)
            {
                if(Sigma_vec[i](j,l) != 0)
                {
                    V_nonzeros_pct++;
                }
                denom++;
            }
        }
    }
    V_nonzeros_pct /= denom;
    r0 = y - X * beta;
    // Eigen::VectorXd& r0, const Eigen::MatrixXd & Z, std::vector<Eigen::MatrixXd> & Sigma_list, 
    // const Eigen::MatrixXi & MAP, int n, int k, int t, double V_nonzeros_pct, int max_itr, 
    // Eigen::MatrixXd & E, Eigen::MatrixXd & D
    converged += estimate_DE(r0,Z,Sigma_vec,MAP,n,k,t,V_nonzeros_pct,max_itr,E,D);
    Rcpp::List Sigma(n); 
    vec2list(Sigma_vec, Sigma);

    return(Rcpp::List::create(Rcpp::Named("Sigma")=Sigma,
           Rcpp::Named("E") = E,Rcpp::Named("D") = D, 
           Rcpp::Named("beta") = beta, 
           Rcpp::Named("converged") = converged));
}
