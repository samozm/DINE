#define STRICT_R_HEADERS
#include "utils.h"

void big_to_small_E(const Eigen::MatrixXd & big_E, Eigen::MatrixXd & E, 
          const Eigen::MatrixXi & MAP, 
          int n, int k, int t)
{
    E = Eigen::MatrixXd::Zero(k,k);
    for(int i=0;i<k;++i)
    {
        E(i,i) = big_E(Eigen::seqN(i*t,t),Eigen::seqN(i*t,t)).diagonal().mean();
    }
    
    for(int i=0;i<k;++i)
    {
        E(i,i) = E(i,i) > 0 ? E(i,i) : double(1) /(n*k);
    }
}

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
        if(idxs.size() != kt0)
        {
            Rcpp::Rcout << "idxs.size = " << idxs.size() << ", kt0 = " << kt0 << "\n";
        }
        R(i,idxs) = r0(Eigen::seqN(current,kt0));
        current += kt0;
    }
    Eigen::MatrixXd master_V = covCalc(R,MAP);
    build_V_list_from_master(V,master_V,MAP,n,k,t);
    return(Rcpp::List::create(Rcpp::Named("V")=V,Rcpp::Named("R")=R,Rcpp::Named("MasterV")=master_V));
}

void estimate_D(const Eigen::VectorXd & r0, const std::vector<Eigen::MatrixXd> & Z,
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
        Eigen::MatrixXd P = ZTZ.colPivHouseholderQr().solve(Z[i].transpose());
        D += P * (V[i] - Et_assemble(E, MAP, i, k, t, V[i].rows())) * P.transpose();
    }
}

Rcpp::List estimate_D(const Eigen::VectorXd & r0, const std::vector<Eigen::MatrixXd> & Z,
                const Eigen::MatrixXd & E, 
                const std::vector<Eigen::MatrixXd> & V, const Eigen::MatrixXi & MAP, 
                int n, int k, int t)
{
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(2*k,2*k);
    estimate_D(r0,Z,D,E,V,MAP,n,k,t);
    return(Rcpp::List::create(Rcpp::Named("D")=D));
}

Rcpp::List estimate_E(const Eigen::VectorXd & r0, 
                      const std::vector<Eigen::MatrixXd> & Z,
                      const Eigen::MatrixXd & D, Eigen::MatrixXd & E, 
                      const std::vector<Eigen::MatrixXd> & V,
                      const Eigen::MatrixXi & MAP, 
                      int n, int k, int t)
{
    // We only need the diagonals of the ZDZ matrix
    std::vector<Eigen::MatrixXd> ZDZ(n); //Z * D * Z.transpose();
    for(int i=0; i<n; ++i)
    {   
        // We only need the diagonals of the ZDZ matrix
        ZDZ[i] = Z[i] * D * Z[i].transpose();
    }
    E = Eigen::MatrixXd::Zero(k,k);
    Eigen::MatrixXd E_tmp = Eigen::MatrixXd::Zero(k,t);
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
                    // Just the most recent time for subject l
                    double B = Vl.coeff(ct(l),ct(l));
                    double C = ZDZl.coeff(ct(l),ct(l));
                    E_tmp(i,j) = E_tmp(i,j) + B - C; //((double) E_tmp(i,j)) + ((double) Vl(i*k+ct(l),i*k+ct(l)) - ZDZ(l*n + i*k + ct(l),l*n + i*k + ct(l));)
                    ct(l)++;
                }
            }
            if(E_tmp(i,j) <= 0)
            {
                E_tmp(i,j) = (1.0)/(n*k);
            }
            else
            {
                E_tmp(i,j) /= bot;
            }
        }

        E(i,i) = E_tmp(i,Eigen::all).mean();
    }
    Rcpp::List ZDZ_list(n);
    vec2list(ZDZ,ZDZ_list);
    return(Rcpp::List::create(Rcpp::Named("E")=E,Rcpp::Named("E")=E,Rcpp::Named("ZDZ")=ZDZ_list));
}


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

void thresholdRange(const Eigen::MatrixXd & R, Eigen::ArrayXXd& theta, Eigen::MatrixXd& cov, const Eigen::MatrixXi & MAP, double & lower, double & upper)
{
    int n = R.rows();
    int p = R.cols();
    cov = covCalc(R,MAP);

    theta = (RtR(R,MAP) - cov.array().pow(2).matrix()).array().sqrt();
    Eigen::MatrixXd delta = (cov.array() / theta).cwiseAbs().matrix();
    delta.diagonal() = Eigen::VectorXd::Zero(p);
    upper = delta.maxCoeff();
    lower = (delta.array() <= 0.f).select(std::numeric_limits<int>::max(), delta).minCoeff();
    /*return(Rcpp::List::create(Rcpp::Named("theta")=theta,
                              Rcpp::Named("upper")=upper,
                              Rcpp::Named("lower")=lower,
                              Rcpp::Named("delta")=delta));*/
}

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

void threshold_V(const Eigen::MatrixXd & R, Eigen::MatrixXd& sigma, 
                 Eigen::ArrayXXd & theta, const Eigen::MatrixXi & MAP, int n_fold=5)
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
            //Eigen::MatrixXd thetalambda = params[j] * thetaTrain;
            Eigen::MatrixXd sigmaTrain(p,p);
            threshold(absCovTrain,signCovTrain,params[j] * thetaTrain,sigmaTrain);
            error(i,j) = (sigmaTrain - covTest).norm();
        }
    }
    Eigen::Index minIndex;
    error.colwise().sum().minCoeff(&minIndex);
    Eigen::ArrayXXd absCov = cov.cwiseAbs();
    Eigen::ArrayXXd signCov = cov.cwiseSign(); // cov.array() / absCov;

    /*Rcpp::Rcout << "err" << "\n";
    Rcpp::Rcout << error.colwise().sum() << "\n";
    Rcpp::Rcout << "theta " << printdims(theta) << "\n";
    Rcpp::Rcout << params[minIndex] * theta(Eigen::seqN(0,5), Eigen::seqN(0,5)) << "\n";

    Rcpp::Rcout << "abscov" << printdims(absCov) << "\n";
    Rcpp::Rcout << absCov(Eigen::seqN(0,5), Eigen::seqN(0,5)) << "\n";*/

    threshold(absCov,signCov,params[minIndex] * theta,sigma);

    //Rcpp::Rcout << "theta" << printdims(theta) << "\n";
    //Rcpp::Rcout << theta(Eigen::seqN(0,5), Eigen::seqN(0,5)) << "\n";
    //return(Rcpp::List::create(Rcpp::Named("V") = sigma));
}

Rcpp::List calc_ZDZ_wrapper(const std::vector<Eigen::MatrixXd>& Z, //Matrix<double, -1,-1,Eigen::RowMajor> & Z, 
                            const Eigen::MatrixXd & D, const Eigen::VectorXd & E,
                            const Eigen::MatrixXi & MAP,
                            int n, int k, int t)
{
    std::vector<Eigen::MatrixXd> out(n);
    calc_ZDZ_plus_E_list(Z,D,E,out,MAP,n,k,t);
    return(Rcpp::List::create(Rcpp::Named("Sigma")=out));
}

Rcpp::List threshold_D(Eigen::MatrixXd & D, double nonzero_pct)
{
    int p = D.rows();
    
    int cutoff = std::round(p*p*nonzero_pct) - 1;
    Eigen::VectorXd DoffVecSorted = D.cwiseAbs().reshaped();
    std::sort(DoffVecSorted.begin(), DoffVecSorted.end(), [](double const& t1, double const& t2){ return t1 < t2; } );
    double threshold = cutoff == 0 ? 0 : DoffVecSorted[cutoff];
    Eigen::MatrixXd thresholdMat = Eigen::MatrixXd::Constant(p,p,1.0) * threshold;
    Eigen::MatrixXd D_tmp = D.cwiseAbs() - thresholdMat;

    /*Rcpp::Rcout << "cutoff = " << cutoff << "\n";
    Rcpp::Rcout << "threshold = " << threshold << "\n";
    Rcpp::Rcout << "threshold = " << thresholdMat << "\n";
    Rcpp::Rcout << "DoffVecSorted = " << DoffVecSorted(Eigen::seqN(0,5)) << "\n";*/

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

void initial_estimates(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, 
                       Eigen::VectorXd & r_out, Eigen::VectorXd & beta, 
                       std::vector<Eigen::MatrixXd> & Sigma_list, 
                       const Eigen::MatrixXi & MAP,
                       int n, int k, int t)
{
    int p = X.cols();
    beta = (X.transpose() * X).llt().solve(X.transpose()) * y;
    r_out = y - X * beta;
    /*Rcpp::Rcout << printdims(r_out) << "\n";
    Rcpp::Rcout << r_out(Eigen::seqN(0,5)) << "\n";*/
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n,k*t);
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

    //Rcpp::Rcout << "initial cov calc next" << "\n";
    Eigen::MatrixXd cov_int = covCalc(R,MAP);
    //Rcpp::Rcout << "build v list from master next" << "\n";
    build_V_list_from_master(Sigma_list, cov_int, MAP, n, k, t);
    /*Rcpp::Rcout << "master V" << printdims(cov_int) << "\n";
    Rcpp::Rcout << cov_int(Eigen::seqN(0,5),Eigen::seqN(0,5)) <<  "\n";
    Rcpp::Rcout << "V" << printdims(Sigma_list[1]) << "\n";
    Rcpp::Rcout << Sigma_list[1](Eigen::seqN(0,5),Eigen::seqN(0,5)) <<  "\n";*/
    
}

Rcpp::List initial_estimates(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, 
                             const Eigen::MatrixXi & MAP,
                             int n, int k, int t, int nkt)//, vector<int> t)
{
    int p = X.cols();
    std::vector<Eigen::MatrixXd> Sigma_list(n);
    Eigen::VectorXd beta(p);
    Eigen::VectorXd r_out(nkt);
    initial_estimates(X,y,r_out,beta,Sigma_list,MAP,n,k,t);
    return(Rcpp::List::create(Rcpp::Named("Sigma_list")=Sigma_list, 
                              Rcpp::Named("beta")=beta,Rcpp::Named("r")=r_out));
}

int estimate_DE(Eigen::VectorXd& r0, const std::vector<Eigen::MatrixXd> & Z, 
                std::vector<Eigen::MatrixXd> & Sigma_list, const Eigen::MatrixXi & MAP, 
                int n, int k, int t, double V_nonzeros_pct, int max_itr, 
                Eigen::MatrixXd & E, Eigen::MatrixXd & D,
                double convergence_cutoff=0.00005)
{
    D = Eigen::MatrixXd::Identity(2*k,2*k) * 0.0005;
    //Eigen::MatrixXd E_tmp = Eigen::MatrixXd::Identity(k,k) * 0.5; 
    E = Eigen::MatrixXd::Identity(k,k) * 0.0005; 
    double err = 10.;
    double prev_err = 10.;
    int n_itr = 0;

    while (((err > convergence_cutoff) || (prev_err > convergence_cutoff)) && (n_itr < max_itr))
    {
        Eigen::MatrixXd D_prev = D;
        Eigen::MatrixXd E_prev = E; //E_tmp;

        /*estimate_D(r0,Z,D,E_tmp,Sigma_list,MAP,n,k,t);
        estimate_E(r0,Z,D,E_tmp,Sigma_list,MAP,n,k,t);*/
        estimate_D(r0,Z,D,E,Sigma_list,MAP,n,k,t);
        estimate_E(r0,Z,D,E,Sigma_list,MAP,n,k,t);

        prev_err = err;
        //err = ((D - D_prev).squaredNorm() / (4*k^2) + (E_tmp - E_prev).squaredNorm() / k) / 2;
        err = ((D - D_prev).squaredNorm() / (4*k^2) + (E - E_prev).squaredNorm() / k) / 2;
        n_itr++;
    }

    //E = Eigen::MatrixXd::Zero(k,k);
    //big_to_small_E(E_tmp, E, MAP, n, k, t);
    
    //Rcpp::Rcout << "D" << printdims(D) << "\n";
    //Rcpp::Rcout << D(Eigen::seqN(0,5),Eigen::seqN(0,5)) <<  "\n";
    //Rcpp::Rcout << "E" << printdims(E) << "\n";
    //Rcpp::Rcout << E(Eigen::seqN(0,5),Eigen::seqN(0,5)) <<  "\n";
    threshold_D(D,V_nonzeros_pct);
    return((n_itr < (max_itr-1)));
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


int estimate_betaV(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, 
                   Eigen::VectorXd & beta, std::vector<Eigen::MatrixXd> & V, 
                   Eigen::MatrixXd & masterV,
                   const Eigen::MatrixXi & MAP, const Eigen::VectorXi kt_vec, 
                   int n, int k, int t, int max_itr,
                   double convergence_cutoff=5*pow(10,-4))
{
    int p = X.cols();
    beta = (X.transpose() * X).colPivHouseholderQr().solve(X.transpose()) * y;
    double err = 10.;
    double prev_err = 10.;
    int n_itr = 0;
    std::vector<double> all_err(max_itr);

    while (((err > convergence_cutoff) || (prev_err > convergence_cutoff)) && (n_itr < max_itr))
    {
        Eigen::VectorXd beta_prev = beta;
        std::vector<Eigen::MatrixXd> V_prev = V;

        estimate_beta(X,y,kt_vec,MAP,V,beta,n,k,t);
        estimate_V(X,y,V,beta,MAP,n,k,t);

        prev_err = err;
        err = ((beta - beta_prev).squaredNorm() / p + (sigma_norm_diff(V,V_prev,n)/ (n*pow(t*k,2)))) / 2;
        all_err[n_itr] = err;
        n_itr++;
    }

    masterV = Eigen::MatrixXd::Zero(t*k,t*k);
    Eigen::ArrayXXd theta = Eigen::ArrayXXd::Zero(2*k,2*k);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n,k*t);
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
    
    threshold_V(R,masterV,theta,MAP);
    build_V_list_from_master(V,masterV,MAP,n,k,t);
    return((n_itr < (max_itr-1)));
}


// [[Rcpp::export]]
Rcpp::List estimate_all(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, 
                        const Rcpp::List & Z_in, int n, int k, int t, 
                        int max_itr=250,
                        double convergence_cutoff=0.00005,
                        bool REML=false)
{
    int p = X.cols();
    std::vector<Eigen::MatrixXd> Sigma_vec(n);
    std::vector<Eigen::MatrixXd> Z(n);
    list2vec(Z,Z_in);
    Rcpp::List Sigma_list(n);
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(2*k,2*k);
    Eigen::MatrixXd E = Eigen::MatrixXd::Zero(k,k);
    Eigen::MatrixXd masterV = Eigen::MatrixXd::Zero(k*t,k*t);
    Eigen::VectorXd beta(p);
    //X, y, Z, r_out, beta, Sigma_list, MAP,
    Eigen::MatrixXd masterZ(k*t,2*k);
    Eigen::MatrixXi MAP = Eigen::MatrixXi::Zero(k*t,2*k);
    Eigen::VectorXd r0;
    
    int nkt = make_MAP(Z,masterZ,MAP,r0,n,k,t);
    
    /*Rcpp::Rcout << "Z0" << "\n"; //<< printdims(Z[0]) << "\n";
    Rcpp::Rcout << Z[0] << "\n";

    Rcpp::Rcout << "MAP(" << MAP.rows() << "," << MAP.cols() << ")" << "\n";
    Rcpp::Rcout << MAP <<  "\n";

    Rcpp::Rcout << "times(" << times.size() << ")" << "\n";
    Rcpp::Rcout << printvec(times) <<  "\n";

    Rcpp::Rcout << "MAP(" << MAP.rows() << "," << MAP.cols() << ")" << "\n";
    Rcpp::Rcout << MAP(Eigen::seqN(0,5),Eigen::seqN(0,20)) <<  "\n";*/

    initial_estimates(X,y,r0,beta,Sigma_vec,MAP,n,k,t);
    //Sigma_list = init_est[0];
    //beta = init_est[1];
    //r0 = init_est[2];
    //Rcpp::Rcout << "4.5" << "\n";
    int converged = estimate_betaV(X,y,beta,Sigma_vec,masterV,MAP,MAP.rowwise().sum(),n,k,t,max_itr,convergence_cutoff);
    //Rcpp::Rcout << "converged = " << converged << "\n";
    double V_nonzeros_pct = 0;
    int denom = 0;
    /*for(int i=0; i<n; ++i)
    {
        int kt0 = Z[i].rows();
        for(int j=0; j<kt0; ++j)
        {
            for(int l=0;l<=j;++l)
            {
                //Rcpp::Rcout << "Sigma [" << i << "] is (" << Sigma_vec[i].rows() << " x " << Sigma_vec[i].cols() << ")\n";
                if(Sigma_vec[i](j,l) != 0)
                {
                    V_nonzeros_pct++;
                }
                denom++;
            }
        }
    }*/
    for(int j=0; j<masterV.rows(); ++j)
    {
        for(int l=0;l<=j;++l)
        {
            if(masterV(j,l) != 0)
            {
                V_nonzeros_pct++;
            }
            denom++;
        }
    }
    //Rcpp::Rcout << "6" << "\n";
    V_nonzeros_pct /= denom;
    //Rcpp::Rcout << "V_nonzeros_pct = " << V_nonzeros_pct << " (denom = " << denom << ")\n";
    r0 = y - X * beta;
    // Eigen::VectorXd& r0, const Eigen::MatrixXd & Z, std::vector<Eigen::MatrixXd> & Sigma_list, 
    // const Eigen::MatrixXi & MAP, int n, int k, int t, double V_nonzeros_pct, int max_itr, 
    // Eigen::MatrixXd & E, Eigen::MatrixXd & D
    converged += 3*estimate_DE(r0,Z,Sigma_vec,MAP,n,k,t,V_nonzeros_pct,max_itr,E,D,convergence_cutoff);
    //Rcpp::Rcout << "converged = " << converged << "\n";
    Rcpp::List Sigma = Rcpp::wrap(Sigma_vec); //(n); 
    //vec2list(Sigma_vec, Sigma);

    //Rcpp::Rcout << "Sigma" << "\n"; //<< printdims(Z[0]) << "\n";
    //Rcpp::Rcout << Sigma[1] << "\n";`
    //Rcpp::Rcout << Sigma_vec[0] << "\n";

    return(Rcpp::List::create(Rcpp::Named("Sigma")=masterV,
           Rcpp::Named("E") = E,Rcpp::Named("D") = D, 
           Rcpp::Named("beta") = beta,
           Rcpp::Named("converged") = converged,
           Rcpp::Named("MAP")= MAP));
}
