#define STRICT_R_HEADERS
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#include "utils.h"
#include <chrono>


void estimate_V(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, 
    Eigen::MatrixXd & masterV, const Eigen::VectorXd & beta,
    const Eigen::MatrixXi & MAP,
    int n, int k, int t, double eigen_threshold=pow(10,-4))
{
    Eigen::VectorXd r0 = y - X * beta;
    Eigen::MatrixXd Rt = Eigen::MatrixXd::Zero(k*t,n);
    int current = 0;
    for(int i = 0; i<n; ++i)
    {
        int kt0 = MAP.rowwise().sum()(i);
        std::vector<int> idxs;
        std::vector<int> waste;
        find_all(MAP(i,Eigen::all),1,idxs,waste);
        if(idxs.size() != kt0)
        {
            Rcpp::Rcout << "idxs.size = " << idxs.size() << ", kt0 = " << kt0 << "\n";
        }
        Rt(idxs,i) = r0(Eigen::seqN(current,kt0));
        current += kt0;
    }
    masterV = covCalc(Rt.transpose(),MAP);

    //Project to Positive Definite Space (Eigenvalue Clipping)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(masterV);
    Eigen::VectorXd evals = es.eigenvalues();
    Eigen::MatrixXd evecs = es.eigenvectors();

    bool needs_projection = false;
    double min_variance = 1e-5; // The smallest variance we will allow

    for(int j = 0; j < evals.size(); ++j) 
    {
        // If the variance is negative or zero, clip it to the minimum
        if(evals(j) < min_variance) 
        {
            evals(j) = min_variance; 
            needs_projection = true;
        }
    }

    // Reconstruct the matrix only if we had to fix it
    if(needs_projection) 
    {
        masterV = evecs * evals.asDiagonal() * evecs.transpose();
    }
}

void estimate_D(const Eigen::VectorXd & r0, 
                const Eigen::MatrixXd & masterZt,
                Eigen::MatrixXd & D, 
                const Eigen::VectorXd & E, 
                const Eigen::MatrixXd & masterV, 
                const Eigen::MatrixXi & MAP, 
                int n, int k, int t)
{
    D = Eigen::MatrixXd::Zero(2*k,2*k);
    Eigen::MatrixXd ZTZ = Eigen::MatrixXd::Zero(2*k,2*k);
    Eigen::MatrixXd Zti(2*k,k*t), Vi(k*t,k*t);
    Eigen::VectorXd Et(k*t);
    Eigen::VectorXi kt_vec = MAP.rowwise().sum();
    for(int i=0; i<n; ++i)
    {   
        int kt = kt_vec(i);
        if (kt == 0) continue;
        Z_assemble_IP(masterZt,Zti,MAP,i,k,t,kt);
        ZTZ += Zti * Zti.transpose();
    }
    for(int i=0; i<n; ++i)
    {
        int kt = kt_vec(i);
        if (kt == 0) continue;
        Z_assemble_IP(masterZt,Zti,MAP,i,k,t,kt);
        Et_assemble_IP(E,Et,MAP,i,k,t,kt);
        V_assemble_IP(masterV,Vi,MAP,i,k,t,kt);
        Vi.diagonal() -= Et;
        Eigen::MatrixXd P = ZTZ.ldlt().solve(Zti);
        D += P * Vi * P.transpose();
    }
}

//[[Rcpp::export]]
Rcpp::List estimate_D(const Eigen::VectorXd & r0, 
                const Eigen::MatrixXd & masterZ,
                const Eigen::VectorXd & E, 
                const Eigen::MatrixXd & masterV, const Eigen::MatrixXi & MAP, 
                int n, int k, int t)
{
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(2*k,2*k);
    Eigen::MatrixXd masterZt = masterZ.transpose();
    estimate_D(r0,masterZt,D,E,masterV,MAP,n,k,t);
    return(Rcpp::List::create(Rcpp::Named("D")=D));
}

void estimate_E(const Eigen::VectorXd & r0, 
                      const Eigen::MatrixXd & masterZt,
                      const Eigen::MatrixXd & D, 
                      Eigen::VectorXd & E, 
                      const Eigen::MatrixXd & masterV,
                      const Eigen::MatrixXi & MAP, 
                      int n, int k, int t)
{
   Eigen::MatrixXd ZDZ = masterZt.transpose() * D * masterZt; // kt x kt
   Eigen::VectorXd Et = masterV.diagonal() - ZDZ.diagonal();
   E = Eigen::VectorXd::Zero(k);
   for(int i = 0; i < k; ++i)
   {
        double sum = 0.0;
        for(int j = 0; j < t; ++j) 
        {
            double sgma_tmp = Et(i*t + j);
            
            // Clamp individual timepoints
            if (sgma_tmp > 0.0) {
                sum += sgma_tmp;
            } else {
                sum += 1.0 / (n * k); 
            }
        }
        
        // Average the clamped values
        E(i) = sum / t;
   }
}

//[[Rcpp::export]]
Rcpp::List estimate_E(const Eigen::VectorXd & r0, const Eigen::MatrixXd & masterZ,
                      const Eigen::MatrixXd & D, 
                      const Eigen::MatrixXd & masterV, 
                      const Eigen::MatrixXi & MAP, 
                      int n, int k, int t)
  
{
    Eigen::VectorXd E(k);
    Eigen::MatrixXd masterZt = masterZ.transpose();
    estimate_E(r0,masterZt,D,E,masterV,MAP,n,k,t);
    return(Rcpp::List::create(Rcpp::Named("E")=E));
}


void thresholdRange(const Eigen::MatrixXd & R, Eigen::ArrayXXd& theta, Eigen::MatrixXd& cov, 
                    const Eigen::MatrixXi & MAP, double & lower, double & upper)
{
    int n = R.rows();
    int p = R.cols();
    cov = covCalc(R,MAP);

    // Prevent NaN from sqrt(negative floating point noise)
    theta = (RtR(R, MAP) - cov.array().square().matrix()).array().max(0.0).sqrt();
    
    Eigen::ArrayXXd safe_theta = (theta == 0.0).select(1e-8, theta);
    Eigen::MatrixXd delta = (cov.array() / safe_theta).cwiseAbs().matrix();
    delta.diagonal() = Eigen::VectorXd::Zero(delta.rows());
    
    upper = delta.maxCoeff();
    lower = (delta.array() <= 0.0).select(std::numeric_limits<double>::max(), delta).minCoeff();
    if (lower == std::numeric_limits<double>::max()) lower = 0.0;
}

void threshold(const Eigen::MatrixXd& abscov, const Eigen::MatrixXd& signcov, double lambda,
               const Eigen::ArrayXXd& theta, Eigen::MatrixXd& sigma_out)
{
    int p = abscov.rows();
    
    // Loop through the matrix without allocating any temporary arrays
    for(int i = 0; i < p; ++i) 
    {
        double diag_diff_i = abscov(i,i);
        for(int j = 0; j < p; ++j) 
        {
            double val = abscov(i, j) - theta(i, j) * lambda;
            sigma_out(i, j) = std::max(0.0, val) * signcov(i, j);
        }
        sigma_out(i, i) = std::max(0.0, diag_diff_i);
    }
}


void threshold_V(const Eigen::MatrixXd & R, Eigen::MatrixXd& sigma, 
                 Eigen::ArrayXXd & theta, const Eigen::MatrixXi & MAP, 
                 int n_fold=5, int seed=11232)
{
    auto rng = std::default_random_engine(seed);

    int n = R.rows();
    int p = R.cols();

    // Dynamic Fold Adjustment for Subsets
    int actual_folds = std::min(n_fold, n);
    
    // Bailing out safely if the subset is extremely small (1 subject)
    if (actual_folds < 2) {
        Eigen::MatrixXd cov = covCalc(R, MAP);
        Eigen::MatrixXd covAbs = cov.cwiseAbs();
        Eigen::MatrixXd covSign = cov.cwiseSign();
        threshold(covAbs, covSign, 1.0, theta, sigma);
        return; 
    }

    int nParam = 100;
    Eigen::MatrixXd cov(p,p);
    double lower = 0.0;
    double upper = 0.0;
    sigma.setZero(p,p);
    
    // 1. Pre-calculate the TOTAL dataset aggregates ONCE
    std::vector<int> all_idx(n);
    std::iota(all_idx.begin(), all_idx.end(), 0);
    
    Eigen::MatrixXd SumXY_tot, N_tot, SumX_shared_tot, SumRsq_tot;
    get_cov_stats(R, MAP, all_idx, SumXY_tot, N_tot, SumX_shared_tot, SumRsq_tot);
    // Build the master baseline cov and theta
    build_cov_and_theta(SumXY_tot, N_tot, SumX_shared_tot, SumRsq_tot, cov, theta);
    get_bounds(cov, theta, lower, upper);

    std::vector<double> params(nParam);
    double jump = (upper - lower)/double(nParam);
    double ctr = lower;
    std::generate(params.begin(), params.end(), [&ctr,&jump]{ return ctr+=jump;});
    
    std::vector<int> part(n);
    std::iota(part.begin(), part.end(), 0);
    std::shuffle(part.begin(),part.end(), rng);

    Eigen::MatrixXd error = Eigen::MatrixXd::Zero(actual_folds,nParam);
    Eigen::MatrixXd covTest;
    Eigen::MatrixXd covTrain(p,p);
    Eigen::ArrayXXd thetaTrain(p,p);

    for (int i=0;i<part.size();++i)
    {
        part[i] = part[i] % actual_folds;
    }
    for (int i=0;i<actual_folds; ++i)
    {
        std::vector<int> val_idx;
        std::vector<int> not_val_idx; // We keep this for find_all, but we NEVER use it!
        find_all(part, i, val_idx, not_val_idx);
        
        // A. Process ONLY the tiny 20% test fold
        Eigen::MatrixXd SumXY_test, N_test, SumX_shared_test, SumRsq_test;
        get_cov_stats(R, MAP, val_idx, SumXY_test, N_test, SumX_shared_test, SumRsq_test);
        
        // B.Derive the 80% train fold instantly via subtraction!
        Eigen::MatrixXd SumXY_train = SumXY_tot - SumXY_test;
        Eigen::MatrixXd N_train = N_tot - N_test;
        Eigen::MatrixXd SumX_shared_train = SumX_shared_tot - SumX_shared_test;
        Eigen::MatrixXd SumRsq_train = SumRsq_tot - SumRsq_test;
        
        // C. Build the matrices
        Eigen::ArrayXXd dummy_thetaTest; // Not used, just needed for the function
        build_cov_and_theta(SumXY_test, N_test, SumX_shared_test, SumRsq_test, covTest, dummy_thetaTest);
        build_cov_and_theta(SumXY_train, N_train, SumX_shared_train, SumRsq_train, covTrain, thetaTrain);
        
        // D. OpenMP Loop
        std::vector<Eigen::MatrixXd> local_sigmas(nParam, Eigen::MatrixXd::Zero(p, p));
        Eigen::MatrixXd covTrainAbs = covTrain.cwiseAbs();
        Eigen::MatrixXd covTrainSign = covTrain.cwiseSign();

        #pragma omp parallel for
        for(int j=0;j<nParam;++j)
        {
            threshold(covTrainAbs,covTrainSign,params[j],thetaTrain,local_sigmas[j]);
            // Calculate the norm manually. (local_sigmas - covTest).norm() 
            // forces Eigen to create a temporary matrix, which would crash R!
            double sq_err = 0.0;
            for(int r = 0; r < p; ++r) {
                for(int c = 0; c < p; ++c) {
                    double diff = local_sigmas[j](r,c) - covTest(r,c);
                    sq_err += diff * diff;
                }
            }
            // Catch NaN or Infinity before it infects the error matrix
            if (std::isnan(sq_err) || std::isinf(sq_err)) 
            {
                // Penalize this threshold heavily so it is never picked as the minimum,
                // but keep it as a finite double so it doesn't break the column sum!
                error(i,j) = std::numeric_limits<double>::max() / 10000.0; 
            } 
            else 
            {
                error(i,j) = std::sqrt(sq_err);
            }
        }
    }
    Eigen::Index minIndex;
    error.colwise().sum().minCoeff(&minIndex);

    Eigen::MatrixXd covAbs = cov.cwiseAbs();
    Eigen::MatrixXd covSign = cov.cwiseSign();
    threshold(covAbs,covSign,params[minIndex],theta,sigma);
    auto t4 = std::chrono::high_resolution_clock::now();
    theta = params[minIndex] * theta;
}

Rcpp::List calc_ZDZ_wrapper(const Eigen::MatrixXd & masterZ, 
                            const Eigen::MatrixXd & D, const Eigen::VectorXd & E,
                            const Eigen::MatrixXi & MAP,
                            int n, int k, int t)
{
    std::vector<Eigen::MatrixXd> out(n);
    calc_ZDZ_plus_E_list(masterZ,D,E,out,MAP,n,k,t);
    return(Rcpp::List::create(Rcpp::Named("Sigma")=out));
}

Rcpp::List threshold_D(Eigen::MatrixXd & D, double nonzero_pct)
{
    int p = D.rows();
    
    int cutoff = std::round(p*p*nonzero_pct) - 1;
    Eigen::VectorXd DoffVecSorted = D.cwiseAbs().reshaped();
    std::sort(DoffVecSorted.begin(), DoffVecSorted.end(), std::greater<double>() );
    double threshold = cutoff <= 0 ? 0 : DoffVecSorted(cutoff);
    Eigen::MatrixXd thresholdMat = Eigen::MatrixXd::Constant(p,p,1.0) * threshold;
    Eigen::MatrixXd D_tmp = D.cwiseAbs() - thresholdMat;

    for(int i=0;i<p; ++i)
    {
        for(int j=0;j<p; ++j)
        {
            if(D_tmp(i,j) < 0)
            {
                D(i,j) = 0;
            }
        }
    }
    return(Rcpp::List::create(Rcpp::Named("D")=D));
}

int initial_estimates(const Eigen::MatrixXd & X, 
                      const Eigen::VectorXd & y, 
                      Eigen::VectorXd & r_out, 
                      Eigen::VectorXd & beta, 
                      Eigen::MatrixXd & masterV, 
                      const Eigen::MatrixXi & MAP,
                      int n, int k, int t)
{
    int p = X.cols();
    beta = (X.transpose() * X).colPivHouseholderQr().solve(X.transpose() * y);
    r_out = y - X * beta;

    Eigen::MatrixXd Rt = Eigen::MatrixXd::Zero(k*t,n);
    Eigen::MatrixXi kt_vec = MAP.rowwise().sum();
    int current = 0;
    for(int i = 0; i<n; ++i)
    {
        int kt = kt_vec(i);
        std::vector<int> idxs;
        std::vector<int> waste;
        find_all(MAP(i,Eigen::all),1,idxs,waste);
        Rt(idxs,i) = r_out(Eigen::seqN(current,kt));
        current += kt;
    }

    masterV = covCalc(Rt.transpose(),MAP);
    //Project to Positive Definite Space (Eigenvalue Clipping)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(masterV);
    Eigen::VectorXd evals = es.eigenvalues();
    Eigen::MatrixXd evecs = es.eigenvectors();

    bool needs_projection = false;
    double min_variance = 1e-5; // The smallest variance we will allow

    for(int j = 0; j < evals.size(); ++j) 
    {
        // If the variance is negative or zero, clip it to the minimum
        if(evals(j) < min_variance) 
        {
            evals(j) = min_variance; 
            needs_projection = true;
        }
    }

    // Reconstruct the matrix only if we had to fix it
    if(needs_projection) 
    {
        masterV = evecs * evals.asDiagonal() * evecs.transpose();
    }
    return current;
    
}

Rcpp::List initial_estimates(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, 
                             const Eigen::MatrixXi & MAP,
                             int n, int k, int t, int nkt)//, vector<int> t)
{
    int p = X.cols();
    std::vector<Eigen::MatrixXd> Sigma_list(n);
    Eigen::MatrixXd masterV(k*t,k*t);
    Eigen::VectorXd beta(p);
    Eigen::VectorXd r_out(nkt);
    initial_estimates(X,y,r_out,beta,masterV,MAP,n,k,t);
    build_V_list_from_master(Sigma_list,masterV,MAP,n,k,t);
    return(Rcpp::List::create(Rcpp::Named("Sigma_list")=Sigma_list, 
                              Rcpp::Named("beta")=beta,Rcpp::Named("r")=r_out));
}

int estimate_DE(Eigen::VectorXd& r0, 
                const Eigen::MatrixXd & masterZt, 
                Eigen::MatrixXd & masterV, 
                const Eigen::MatrixXi & MAP, 
                int n, int k, int t, 
                double V_nonzeros_pct, int max_itr, 
                Eigen::VectorXd & E, Eigen::MatrixXd & D,
                double convergence_cutoff=0.00005,
                bool timings=false)
{
    D = Eigen::MatrixXd::Identity(2*k,2*k) * 0.0005;
    E = Eigen::VectorXd::Constant(k, 1.0);
    double err = 10.;
    double prev_err = 10.;
    int n_itr = 0;

    while (((err > convergence_cutoff) || (prev_err > convergence_cutoff)) && (n_itr < max_itr))
    {
        // Give control back to R to check for Esc/Ctrl+C
        Rcpp::checkUserInterrupt();

        Eigen::MatrixXd D_prev = D;
        Eigen::VectorXd E_prev = E; //E_tmp;

        auto t1 = std::chrono::high_resolution_clock::now();
        estimate_D(r0,masterZt,D,E,masterV,MAP,n,k,t);
        auto t2 = std::chrono::high_resolution_clock::now();
        estimate_E(r0,masterZt,D,E,masterV,MAP,n,k,t);
        auto t3 = std::chrono::high_resolution_clock::now();

         if(timings)
        {
            double time_D = std::chrono::duration<double, std::milli>(t2 - t1).count();
            double time_E = std::chrono::duration<double, std::milli>(t3 - t2).count();
            
            Rcpp::Rcout << "--- Iteration  " << n_itr << " Timings (ms) ---" << std::endl;
            Rcpp::Rcout << "estimate_D:    " << time_D << " ms" << std::endl;
            Rcpp::Rcout << "estimate_E:    " << time_E << " ms" << std::endl;
        }

        prev_err = err;
        err = ((D - D_prev).squaredNorm() / (D_prev.squaredNorm()) + (E - E_prev).squaredNorm() / E_prev.squaredNorm()) / 2;
        n_itr++;
    }

    threshold_D(D,V_nonzeros_pct);

    return((n_itr < (max_itr-1)));
}


int estimate_betaV(const Eigen::MatrixXd & X, 
                   const Eigen::VectorXd & y, 
                   Eigen::VectorXd & beta, 
                   Eigen::MatrixXd & masterV,
                   const Eigen::MatrixXi & MAP, const Eigen::VectorXi kt_vec, 
                   int n, int k, int t, int max_itr,
                   int n_fold=5, int seed=1121,
                   double convergence_cutoff=5*pow(10,-4),
                   bool timings=false,
                   bool verbose=false)
{
    int p = X.cols();
    beta = (X.transpose() * X).colPivHouseholderQr().solve(X.transpose()) * y;
    double err = 10.;
    double prev_err = 10.;
    int n_itr = 0;
    std::vector<double> all_err(max_itr);

    while (((err > convergence_cutoff) || (prev_err > convergence_cutoff)) && (n_itr < max_itr))
    {
        // Give control back to R to check for Esc/Ctrl+C
        Rcpp::checkUserInterrupt();

        Eigen::VectorXd beta_prev = beta;
        Eigen::MatrixXd masterV_prev = masterV;

        auto t1 = std::chrono::high_resolution_clock::now();
        estimate_beta(X,y,kt_vec,MAP,masterV,beta,n,k,t,verbose);
        auto t2 = std::chrono::high_resolution_clock::now();
        estimate_V(X,y,masterV,beta,MAP,n,k,t);
        auto t3 = std::chrono::high_resolution_clock::now();

        if(timings)
        {
            double time_beta = std::chrono::duration<double, std::milli>(t2 - t1).count();
            double time_V    = std::chrono::duration<double, std::milli>(t3 - t2).count();
            
            Rcpp::Rcout << "--- Iteration   " << n_itr << " Timings (ms) ---" << std::endl;
            Rcpp::Rcout << "estimate_beta:  " << time_beta << " ms" << std::endl;
            Rcpp::Rcout << "estimate_V:     " << time_V << " ms" << std::endl;
        }
        prev_err = err;
        double eps = 1e-8;
        err = ((beta - beta_prev).squaredNorm() / (beta_prev.squaredNorm() + eps) + (masterV - masterV_prev).squaredNorm() / (masterV_prev.squaredNorm() + eps))/2;
        all_err[n_itr] = err;
        if(verbose)
        {
            Rcpp::Rcout << "Iteration " << n_itr << ": err = " << err << ", prev_err = " << prev_err << std::endl;
            Rcpp::Rcout << "beta" << beta.transpose() << std::endl;
            Rcpp::Rcout << "masterV" << masterV(Eigen::seqN(0,5),Eigen::seqN(0,5)) << std::endl;
        }

        n_itr++;
    }

    Eigen::ArrayXXd theta = Eigen::ArrayXXd::Zero(2*k,2*k);
    Eigen::MatrixXd Rt = Eigen::MatrixXd::Zero(k*t,n);
    Eigen::VectorXd r0 = y - X * beta;
    int current = 0;
    for(int i = 0; i<n; ++i)
    {
      int kt = kt_vec(i);
      if(kt == 0) continue;
      std::vector<int> idxs;
      std::vector<int> waste;
      find_all(MAP(i,Eigen::all),1,idxs,waste);
      Rt(idxs,i) = r0(Eigen::seqN(current,kt));
      current += kt;
    }
    
    threshold_V(Rt.transpose(),masterV,theta,MAP,n_fold,seed);
    return((n_itr < (max_itr-1)));
}


// [[Rcpp::export]]
Rcpp::List estimate_all(const Eigen::MatrixXd & X, 
                        const Eigen::VectorXd & y, 
                        const Eigen::MatrixXd & masterZ, 
                        const Eigen::MatrixXi & MAP,
                        int n, int k, int t, 
                        int max_itr=250,
                        double convergence_cutoff=0.00005,
                        bool REML=false,
                        int n_fold=5, 
                        bool timings=false,
                        bool verbose=false,
                        int seed=1121)
{
    int p = X.cols();
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(2*k,2*k);
    Eigen::VectorXd E = Eigen::VectorXd::Zero(k);
    Eigen::MatrixXd masterV = Eigen::MatrixXd::Zero(k*t,k*t);
    Eigen::VectorXd beta(p);
    //X, y, Z, r_out, beta, Sigma_list, MAP,
    Eigen::MatrixXd masterZt = masterZ.transpose();
    Eigen::VectorXd r0;
    

    auto t0 = std::chrono::high_resolution_clock::now();
    int nkt = initial_estimates(X,y,r0,beta,masterV,MAP,n,k,t);
    auto t1 = std::chrono::high_resolution_clock::now();
    int converged = estimate_betaV(X,y,beta,masterV,MAP,MAP.rowwise().sum(),n,k,t,max_itr,n_fold,seed,convergence_cutoff,timings,verbose);
    auto t2 = std::chrono::high_resolution_clock::now();
    double V_nonzeros_pct = 0;
    int denom = 0;
   
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
    V_nonzeros_pct /= denom;
    r0 = y - X * beta;
    auto t3 = std::chrono::high_resolution_clock::now();
    converged += 3*estimate_DE(r0,masterZt,masterV,MAP,n,k,t,V_nonzeros_pct,max_itr,E,D,convergence_cutoff,timings);
    auto t4 = std::chrono::high_resolution_clock::now();

    if(timings)
    {
        double time_init = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double time_betaV = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double time_DE    = std::chrono::duration<double, std::milli>(t4 - t3).count();
        
        Rcpp::Rcout << "------ Overall Timings (ms) ------" << std::endl;
        Rcpp::Rcout << "estimate_init:  " << time_init << " ms" << std::endl;
        Rcpp::Rcout << "estimate_betaV: " << time_betaV << " ms" << std::endl;
        Rcpp::Rcout << "estimate_DE:    " << time_DE << " ms" << std::endl;
    }

    return(Rcpp::List::create(Rcpp::Named("Sigma")=masterV,
           Rcpp::Named("E") = E,Rcpp::Named("D") = D, 
           Rcpp::Named("beta") = beta,
           Rcpp::Named("converged") = converged,
           Rcpp::Named("MAP")= MAP,
           Rcpp::Named("V_nonzeros_pct") = V_nonzeros_pct));
}
