#define STRICT_R_HEADERS
#include "utils.h"
#include <chrono>

/*  
******************************************************************************
******************************************************************************
************ EVERY SUBJECT MUST HAVE ALL NODES AT AT LEAST 1 TIME ************
************                 OR AT LEAST A COLUMN                 ************
************         (ALL MATRICES NEED TO HAVE WIDTH 2K)         ************
******************************************************************************
******************************************************************************
*/

Eigen::VectorXd Zbcalc(const Eigen::MatrixXd & Z, 
                       const Eigen::VectorXd & b, 
                       const Eigen::MatrixXi & MAP, 
                       int n, int k, int t, 
                       int nkt)
{
    Eigen::VectorXd Zb = Eigen::VectorXd::Zero(nkt);
    Eigen::VectorXi kt_vec = MAP.rowwise().sum();
    Eigen::MatrixXd Zi(k*t,2*k);
    int cnt = 0;
    for(int i = 0; i < n; ++i)
    {
        int kt = kt_vec(i);
        Z_assemble_IP(Z,Zi,MAP,i,k,t,kt);
        Zb.segment(cnt,kt) = Zi * b; 
        cnt += kt;
    }
    return(Zb);
}


void calc_b(const Eigen::MatrixXd & X, const Eigen::VectorXd & r0, 
            const Eigen::MatrixXd & Z,
            const Eigen::MatrixXd & Lambda_D, const Eigen::VectorXd & E, Eigen::VectorXd & b,
            const Eigen::MatrixXi & MAP,
            int n, int k, int t, int nkt)
{
    int q = 2*k;
    Eigen::MatrixXd ZEEZ = Eigen::MatrixXd::Zero(q,q);
    //Eigen::MatrixXd DZETE = Eigen::MatrixXd::Zero(q,nkt);
    Eigen::VectorXd DZETEr0 = Eigen::VectorXd::Zero(q);
    Eigen::VectorXd EInv = E.array().inverse(); //E.inverse();
    Eigen::VectorXi kt_vec = MAP.rowwise().sum();
    Eigen::MatrixXd Zi(k*t,q), EtInv(k*t,k*t), EZ(k*t,q);
    Eigen::VectorXd EtInvr0(k*t), ZiEtInvr0(k*t);
    int cnt = 0;
    for(int i=0;i<n;++i)
    {
        int kt = kt_vec(i);
        Z_assemble_IP(Z,Zi,MAP,i,k,t,kt);
        Et_assemble_IP(EInv,EtInv, MAP, i, k, t, kt);
        EZ.resize(kt,q);
        EtInvr0.resize(kt);
        ZiEtInvr0.resize(q);

        EZ.noalias() = EtInv * Zi; 
        ZEEZ.noalias() += EZ.transpose() * EZ;
        EtInvr0.noalias() = EtInv.array().square().matrix() * r0.segment(cnt,kt);
        ZiEtInvr0.noalias() = Zi.transpose() * EtInvr0;
        DZETEr0.noalias() += Lambda_D.transpose() * ZiEtInvr0;
        //DZETE(Eigen::all,Eigen::seqN(cnt,kt)).noalias() = Lambda_D.transpose() * Zi.transpose() * EtInv.array().square().matrix();
        cnt += kt;
    }

    Eigen::MatrixXd DtZEEZD = Eigen::MatrixXd::Identity(q,q);
    DtZEEZD.noalias() += Lambda_D.transpose() * ZEEZ * Lambda_D;
    Eigen::MatrixXd DtZEEZDDZETEr0 = DtZEEZD.llt().solve(DZETEr0);
    b = Lambda_D.transpose() * DtZEEZDDZETEr0;
}

void estimate_E(const Eigen::MatrixXd & X, const Eigen::VectorXd & r0, 
                const Eigen::MatrixXd & Z,
                const Eigen::MatrixXd & Lambda_D, Eigen::VectorXd & Lambda_E,
                const Eigen::MatrixXi & MAP, 
                int n, int k, int t, int nkt)
{
    Eigen::VectorXd b;
    calc_b(X,r0,Z,Lambda_D,Lambda_E,b,MAP,n,k,t,nkt);
    Eigen::VectorXd r = r0 - Zbcalc(Z, b, MAP, n, k, t, nkt);
    //Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n*t,k);
    Eigen::VectorXd sum_val = Eigen::VectorXd::Zero(k);
    Eigen::VectorXd sum_sq = Eigen::VectorXd::Zero(k);
    Eigen::VectorXi counts = Eigen::VectorXi::Zero(k);
    int current = 0;
    for(int i=0; i<n; ++i)
    {
        for(int j=0; j<k; ++j)
        {
            for(int l=0; l<t; ++l)
            {
                if (MAP(i,j*t+l) == 1)
                {
                    //R(i*t + l,j) = r(current);
                    double val = r(current++);
                    sum_val(j) += val;
                    sum_sq(j) += val*val;
                    counts(j)++;
                }
            }
        }
    }
    Eigen::ArrayXd means = sum_val.array() / counts.array().cast<double>();
    Eigen::ArrayXd mean_sq = sum_sq.array() / counts.array().cast<double>();
    
    // Update Lambda_E (standard deviation)
    Lambda_E = (mean_sq - means.square()).max(1e-8).sqrt();
}

void calc_e(const Eigen::VectorXd & r0, const Eigen::MatrixXd & Z,
            const Eigen::VectorXd & E, const Eigen::MatrixXd & Lambda_D, 
            const Eigen::MatrixXi & MAP, Eigen::VectorXd & e, 
            int n, int k, int t, int nkt)
{
    int p = 2*k;
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(p,p);
    Eigen::VectorXi kt_vec = MAP.rowwise().sum();
    Eigen::MatrixXd ZEEZ = Eigen::MatrixXd::Zero(p,p);
    //Eigen::MatrixXd EZ = Eigen::MatrixXd::Zero(nkt,p);
    Eigen::MatrixXd Zi(k*t,p), Et(k*t,k*t), E_tmp(k*t,k*t), ZiT(p,k*t), EZ_tmp(k*t,p);
    Eigen::VectorXd Zr = Eigen::VectorXd::Zero(p);

    int cnt = 0;
    for(int i=0;i<n;++i)
    {
        int kt = kt_vec(i);
        Z_assemble_IP(Z,Zi,MAP,i,k,t,kt);
        ZiT = Zi.transpose();
        B.noalias() += ZiT * Zi;
        Et_assemble_IP(E, E_tmp, MAP, i, k, t, kt);//Z[i];
        EZ_tmp = E_tmp*Zi;
        ZEEZ.noalias() += EZ_tmp.transpose() * EZ_tmp;
        Zr.noalias() += ZiT * r0.segment(cnt,kt);
        cnt += kt;
    }
    Eigen::MatrixXd BD = B * Lambda_D;
    Eigen::MatrixXd BDBinv =  BD * BD.transpose();
    Eigen::MatrixXd BDBinvZEEZ =  BDBinv + ZEEZ;
    Eigen::LLT<Eigen::MatrixXd> BDBinvZEEZ_llt = BDBinvZEEZ.llt();
    //Eigen::MatrixXd HZEEZ(k*t,p);
    Eigen::MatrixXd EZi(k*t,p);
    Eigen::LDLT<Eigen::MatrixXd> BDBinv_ldlt = BDBinv.ldlt();
    Eigen::VectorXd BDBinvZr = BDBinv_ldlt.solve(Zr);
    //Eigen::VectorXd EZiHZEEZBDBinvZr(p);
    Eigen::MatrixXd ZEEZsolve = BDBinvZEEZ_llt.solve(ZEEZ);
    Eigen::VectorXd BDBinvZrZEEZsolveBDBinvZr = BDBinvZr - (ZEEZsolve * BDBinvZr);
    cnt = 0; 
    for(int i=0;i<n;++i)
    {
        int kt = kt_vec(i);
        Et_assemble_IP(E, Et, MAP, i, k, t, kt);
        Z_assemble_IP(Z,Zi,MAP,i,k,t,kt);
        EZi.resize(kt, p);
        //HZEEZ.resize(kt, p);
        EZi.noalias() = Et * Zi;
        //HZEEZ.noalias() = EZi * ZEEZsolve;
        //Eigen::VectorXd EZiBDB = EZi * BDBinvZr;
        //Eigen::VectorXd HZEEZBDB = HZEEZ * BDBinvZr;
        //EZiHZEEZBDBinvZr = EZi 
        e.segment(cnt,kt).noalias() = Et * (EZi * BDBinvZrZEEZsolveBDBinvZr);
        cnt += kt;
    }
}


void a2_thresholdRange(const Eigen::MatrixXd & R, Eigen::ArrayXXd& theta, Eigen::MatrixXd& cov, 
                    const Eigen::MatrixXi & MAP, double & lower, double & upper)
{
    int n = R.rows();
    int p = R.cols();
    cov = covCalc(R,MAP);

    // Prevent NaN from sqrt(negative floating point noise)
    theta = (RtR(R, MAP) - cov.array().square().matrix()).array().max(0.0).sqrt();
    
    // The Subset Safety Net: Prevent Division by Zero!
    Eigen::ArrayXXd safe_theta = (theta == 0.0).select(1e-8, theta);
    Eigen::MatrixXd delta = (cov.array() / safe_theta).cwiseAbs().matrix();
    delta.diagonal() = Eigen::VectorXd::Zero(delta.rows());
    upper = delta.maxCoeff();
    // Fallback in case all delta values are 0 (perfect sparsity)
    lower = (delta.array() <= 0.0).select(std::numeric_limits<double>::max(), delta).minCoeff();
    if (lower == std::numeric_limits<double>::max()) {
        lower = 0.0; 
    }
}

void a2_threshold(const Eigen::MatrixXd& abscov, const Eigen::MatrixXd& signcov, double lambda,
               const Eigen::ArrayXXd& theta, Eigen::MatrixXd& sigma_out)
{
    int p = abscov.rows();
    
    // Loop through the matrix without allocating any temporary arrays
    for(int i = 0; i < p; ++i) 
    {
        double diag_diff_i = abscov(i,i) - (theta(i,i) * lambda);
        for(int j = 0; j < p; ++j) 
        {
            double diag_diff_j = abscov(j,j) - (theta(j,j) * lambda);
            // Apply the diagonal mask and threshold in one step
            if (diag_diff_i > 0.0 && diag_diff_j > 0.0) 
            {
                double val = abscov(i, j) - theta(i, j) * lambda;
                sigma_out(i, j) = std::max(0.0, val) * signcov(i, j);
            } 
            else 
            {
                sigma_out(i, j) = 0.0;
            }
        }
        sigma_out(i, i) = std::max(0.0, diag_diff_i);
    }
}

void a2_threshold_D(const Eigen::MatrixXd & R, Eigen::MatrixXd& sigma, 
                 Eigen::ArrayXXd & theta, const Eigen::MatrixXi & MAP, 
                 int n_fold=5,int seed=1234)
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
        a2_threshold(covAbs, covSign, 1.0, theta, sigma);
        return; 
    }

    int nParam = 100;
    Eigen::MatrixXd cov(p,p);
    double lower = 0.0;
    double upper = 0.0;
    sigma.setZero(p,p);
    
    a2_thresholdRange(R,theta,cov,MAP,lower,upper);
    std::vector<double> params(nParam);
    double jump = (upper - lower)/double(nParam);
    double ctr = lower;
    std::generate(params.begin(), params.end(), [&ctr,&jump]{ return ctr+=jump;});
    
    std::vector<int> part(n);
    std::iota(part.begin(), part.end(), 0);
    std::shuffle(part.begin(),part.end(), rng);

    Eigen::MatrixXd error(n_fold,nParam);
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
        std::vector<int> not_val_idx;
        find_all(part,i,val_idx,not_val_idx);
        double lower = 0.0;
        double upper = 0.0; 
        a2_thresholdRange(R(not_val_idx,Eigen::all), thetaTrain, covTrain, MAP(not_val_idx,Eigen::all), lower, upper);
        covTest = covCalc(R(val_idx,Eigen::all),MAP(val_idx,Eigen::all));
        std::vector<Eigen::MatrixXd> local_sigmas(nParam, Eigen::MatrixXd::Zero(p, p));
        Eigen::MatrixXd covTrainAbs = covTrain.cwiseAbs();
        Eigen::MatrixXd covTrainSign = covTrain.cwiseSign();
        #pragma omp parallel for
        for(int j=0;j<nParam;++j)
        {
            a2_threshold(covTrainAbs,covTrainSign,params[j],thetaTrain,local_sigmas[j]);
            // Calculate the norm manually. (local_sigmas - covTest).norm() 
            // forces Eigen to create a temporary matrix, which would crash R!
            double sq_err = 0.0;
            for(int r = 0; r < p; ++r) {
                for(int c = 0; c < p; ++c) {
                    double diff = local_sigmas[j](r,c) - covTest(r,c);
                    sq_err += diff * diff;
                }
            }
            error(i,j) = std::sqrt(sq_err);
        }
    }
    Eigen::Index minIndex;
    error.colwise().sum().minCoeff(&minIndex);
    //theta = params[minIndex] * theta;

    Eigen::MatrixXd covAbs = cov.cwiseAbs();
    Eigen::MatrixXd covSign = cov.cwiseSign();
    a2_threshold(covAbs,covSign,params[minIndex],theta,sigma);
    theta = params[minIndex] * theta;

}

void estimate_D(const Eigen::MatrixXd & X, const Eigen::VectorXd & r0, 
                const Eigen::MatrixXd & Z,
                const Eigen::VectorXd & E, Eigen::MatrixXd & Lambda_D, 
                const Eigen::MatrixXi & MAP, Eigen::MatrixXd & D,
                Eigen::ArrayXXd & theta,
                int n, int k, int t, int nkt, 
                int itr, int n_fold=5, 
                bool custom_theta=false, 
                int seed=1111,
                bool soft=1, 
                double eigen_threshold=pow(10,-2))
{
    int p = 2*k;
    Eigen::VectorXd e = Eigen::VectorXd::Zero(nkt);
    calc_e(r0,Z,E,Lambda_D,MAP,e,n,k,t,nkt);
    Eigen::MatrixXd R(n,p),Zi(k*t,p),ZiTZi(p,p); // = Eigen::MatrixXd::Zero(n,p);
    Eigen::VectorXd ZiTr(p);
    Eigen::VectorXd r = r0 - e;
    int cnt = 0;
    Eigen::VectorXi kt_vec = MAP.rowwise().sum();
    for(int i=0;i<n;++i)
    {
        int kt = kt_vec(i);//Z[i].rows();
        Z_assemble_IP(Z,Zi,MAP,i,k,t,kt);
        ZiTZi.noalias() = Zi.transpose() * Zi;
        ZiTZi.diagonal().array() += 1e-8; // ridge in case a whole column is 0
        ZiTr.noalias() = Zi.transpose() * r.segment(cnt,kt);
        Eigen::LLT<Eigen::MatrixXd> llt(ZiTZi);
        R.row(i) = llt.solve(ZiTr).transpose(); // solve returns a column vector
        cnt += kt;
    }

    //TODO: only change theta every 5 iterations???
    //give option to have user input threshold theta
    if(itr % 5 == 0 && !custom_theta)
    {
        a2_threshold_D(R,D,theta,MAP,n_fold,seed=seed);
    }else{
        Eigen::MatrixXd cov = covCalc(R, MAP);
        D.setZero(p,p);
        //TODO: sqrt log p/n??
        a2_threshold(cov.cwiseAbs(),cov.cwiseSign(),1.0,theta,D);
    }


    Eigen::MatrixXd solver_input = D + Eigen::MatrixXd::Identity(p, p);

    // 2. Attempt Cholesky Decomposition (LLT) directly
    Eigen::LLT<Eigen::MatrixXd> llt(solver_input);
    
    // 3. If it fails (matrix is not Positive Definite), apply an iterative ridge
    if(llt.info() == Eigen::NumericalIssue)
    {
        // Start with your chosen threshold
        double current_shift = eigen_threshold; 
        
        while(llt.info() == Eigen::NumericalIssue)
        {
            // Push the diagonal up
            solver_input.diagonal().array() += current_shift;
            D.diagonal().array() += current_shift;
            
            // Re-attempt the decomposition
            llt.compute(solver_input);
            
            // Aggressively increase the shift in case the matrix is highly negative
            current_shift *= 2.0; 
        }
    }
    
    // 4. We are now guaranteed a valid Lower Triangular matrix
    Lambda_D = llt.matrixL();
}

double calc_sigma2(const Eigen::MatrixXd & Z, 
                   const Eigen::MatrixXd& D, const Eigen::VectorXd& E, 
                   const Eigen::MatrixXi & MAP,
                   const Eigen::VectorXd& r0, 
                   int n, int k, int t, int p, 
                   bool REML=false)
{
    double sigma2 = 0.0;
    Eigen::MatrixXd Lambda_V;
    int nkt = 0;
    Eigen::VectorXi kt_vec = MAP.rowwise().sum();
    Eigen::MatrixXd Zi(k*t,2*k),Et(k*t,k*t), ZDZit(k*t,k*t), ZiD(k*t,2*k);
    for(int i=0; i<n;++i)
    {
        int kt = kt_vec(i);
        Et_assemble_IP(E, Et, MAP, i, k, t, kt);
        Z_assemble_IP(Z,Zi,MAP,i,k,t,kt);

        ZiD.resize(kt, 2*k);
        ZDZit.resize(kt, kt);
        // broken up for stupid compiler reasons
        ZiD.noalias() = Zi * D;
        ZDZit.noalias() = ZiD * Zi.transpose();
        ZDZit += Et;

        Lambda_V = ZDZit.llt().matrixL();
        sigma2 += (Lambda_V.triangularView<Eigen::Lower>().solve(r0.segment(nkt,kt))).squaredNorm();//(Lambda_V.colPivHouseholderQr().solve(r0(Eigen::seqN(nkt,kt)))).squaredNorm();
        nkt += kt;
    }
    if(REML)
    {
        sigma2 = sigma2 / (nkt- p);
    } else {
        sigma2 = sigma2 / (nkt);
    }
    return(sigma2);
}

int a2_initial_estimates(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, 
                       const Eigen::MatrixXd & Z, 
                       const Eigen::MatrixXi & MAP, 
                       Eigen::MatrixXd & Lambda_D, Eigen::MatrixXd & D, 
                       Eigen::VectorXd & Lambda_E, Eigen::VectorXd & beta, 
                       Eigen::VectorXd & r,
                       int n, int k, int t)
{
    int p = X.cols();
    int q = 2*k; 
    Eigen::MatrixXd XtX = X.transpose() * X;
    Eigen::MatrixXd XtXinvXt = (XtX).ldlt().solve(X.transpose());
    beta = XtXinvXt * y;
    r = y - X * beta;
    Eigen::MatrixXd R(n,k*t);
    int nkt = 0;
    for(int i = 0; i<n; ++i)
    {
        std::vector<int> idxs;
        std::vector<int> waste;
        int kt0 = MAP.rowwise().sum()(i);
        find_all(MAP(i,Eigen::all),1,idxs,waste);
        R(i,idxs) = r.segment(nkt,kt0);
        nkt += kt0;
    }
    Eigen::MatrixXd cov_int = covCalc(R, MAP);
    for(int i=0;i<k;++i)
    {
        Lambda_E(i) = (cov_int.diagonal().array()).segment(i*t,t).sqrt().mean() /2;
    }

    cov_int.diagonal() = cov_int.diagonal()/2;

    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(q,q); 
    Eigen::MatrixXd ZCZ = Eigen::MatrixXd::Zero(q,q);
    Eigen::VectorXi kt_vec = MAP.rowwise().sum();
    Eigen::MatrixXd Zi(k*t,2*k);
    for(int i=0; i<n;++i)
    {
        int kt = kt_vec(i);
        Z_assemble_IP(Z,Zi,MAP,i,k,t,kt);
        B += Zi.transpose() * Zi;
        ZCZ += Zi.transpose() * cov_int * Zi;
    }
    // 1. Solve the first half: M = B^-1 * ZCZ
    Eigen::MatrixXd M = B.ldlt().solve(ZCZ);
    
    // 2. Solve the second half: B * D^T = M^T
    Eigen::MatrixXd D_transpose = B.ldlt().solve(M.transpose());
    
    // 3. Transpose back to get D
    D = D_transpose.transpose();
    Lambda_D = (D + Eigen::MatrixXd::Identity(q,q)).llt().matrixL();
    return nkt;
}


//' Estimate Covariance Matrices and Fixed Effects (C++ Backend)
//'
//' @description A high-performance, OpenMP-enabled C++ backend for iteratively estimating 
//' fixed effects (\code{beta}), random effect covariance (\code{D}), and residual variance (\code{E}) 
//' using block-coordinate descent and cross-validated thresholding.
//'
//' @param X Numeric matrix of fixed effect covariates.
//' @param y Numeric vector of the continuous response variable.
//' @param masterZ Master random effect matrix containing all timepoints used by any subject
//' @param MAP Integer matrix, n x kt describing which node/timepoint combinations each subject has
//' @param n Integer. Number of subjects.
//' @param k Integer. Number of nodes or spatial features.
//' @param t Integer. Number of time points per subject.
//' @param theta Numeric matrix. Initial thresholding parameters for cross-validation.
//' @param max_itr Integer. Maximum number of block-coordinate descent iterations. Default is 250.
//' @param convergence_cutoff Numeric. Change in error required to declare convergence. Default is 0.0001.
//' @param REML Logical. If TRUE, uses Restricted Maximum Likelihood for variance estimation. Default is FALSE.
//' @param verbose Logical. If TRUE, prints iteration-level errors and matrix updates to the console. Default is FALSE.
//' @param timings Logical. If TRUE, prints millisecond timings for internal C++ functions. Default is FALSE.
//' @param n_fold Integer. Number of folds for the internal cross-validation step. Default is 5.
//' @param custom_theta Logical. If TRUE, bypasses cross-validation and uses the user-provided \code{theta}. Default is FALSE.
//' @param n_threads Integer. Number of OpenMP threads to use. Set to 1 if parallelizing at the R level. Default is 1.
//' @param seed Integer. Random seed for ensuring reproducible cross-validation splits. Default is 1234.
//' 
//' @return A named list containing:
//' \itemize{
//'   \item \code{Sigma}: The final composite covariance matrix.
//'   \item \code{E}: The diagonal residual variance matrix.
//'   \item \code{D}: The thresholded random effects covariance matrix.
//'   \item \code{Lambda_E}: The standard deviation vector of the residuals.
//'   \item \code{beta}: The final fixed-effects coefficient vector.
//'   \item \code{n_iter}: The total number of iterations run.
//'   \item \code{all_err}: Vector of convergence errors across all iterations.
//'   \item \code{converged}: Logical indicating if the algorithm reached \code{convergence_cutoff} before \code{max_itr}.
//'   \item \code{sigma}: The estimated scaling variance parameter.
//'   \item \code{threshold}: The final threshold matrix selected by cross-validation.
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List estimate_DEbeta(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, 
                           const Eigen::MatrixXd & masterZ,
                           const Eigen::MatrixXi & MAP,
                           int n, int k, int t,
                           Eigen::ArrayXXd theta,
                           int max_itr=250, 
                           double convergence_cutoff=0.0001,
                           bool REML=false,
                           bool verbose=false,
                           bool timings=false,
                           int n_fold=5,
                           bool custom_theta = false,
                           int n_threads = 1,
                           int seed=1234)
{
    // Protect C++ from R's NAs
    // TODO: better NA handling?
    if (X.hasNaN() || y.hasNaN() || masterZ.hasNaN()) {
        Rcpp::stop("Input matrix X or vector y contains NA/NaN values. Please remove them before running DINE.");
    }

    // Lock Eigen to 1 thread permanently 
    Eigen::setNbThreads(1); 
    
    // Control OpenMP dynamically from R!
    #ifdef _OPENMP
        omp_set_num_threads(n_threads);
    #endif

    int p = X.cols();
    Eigen::MatrixXd Lambda_D(2*k,2*k);
    Eigen::MatrixXd D(2*k,2*k);
    Eigen::VectorXd Lambda_E(k);
    Eigen::VectorXd beta(p);
    Eigen::VectorXd r0;
    
    int nkt = a2_initial_estimates(X,y,masterZ,MAP,Lambda_D,D,Lambda_E,beta,r0,n,k,t);

    Eigen::VectorXi kt_vec = MAP.rowwise().sum();
    Eigen::VectorXd beta_prev;
    Eigen::MatrixXd D_prev;
    Eigen::VectorXd E_prev;
    
    double sigma2 = 0.0;
    double err = 10.0;
    double prev_err = 9.0;
    double double_prev_err = 8.0;
    int n_itr = 0;
    std::vector<double> all_err(max_itr);
    while (((err > convergence_cutoff) || (prev_err > convergence_cutoff)) && (n_itr < max_itr))
    {
        // Give control back to R to check for Esc/Ctrl+C
        Rcpp::checkUserInterrupt();

        auto start_loop = std::chrono::high_resolution_clock::now();
        double_prev_err = prev_err;
        prev_err = err;
        beta_prev = beta;
        auto t1 = std::chrono::high_resolution_clock::now();
        estimate_beta2(X,y,masterZ,D,Lambda_E.array().square(),kt_vec,MAP,beta,n,k,t);
        auto t2 = std::chrono::high_resolution_clock::now();
        double err2 = (beta - beta_prev).squaredNorm() / beta_prev.squaredNorm(); 

        r0 = y - X * beta; 
        D_prev = Lambda_D;
        estimate_D(X,r0,masterZ,Lambda_E.array().square(),Lambda_D,MAP,D,theta,n,k,t,nkt,n_itr,n_fold,custom_theta);
        auto t3 = std::chrono::high_resolution_clock::now();
        double err0 = (Lambda_D - D_prev).squaredNorm() / D_prev.squaredNorm();

        E_prev = Lambda_E;
        estimate_E(X,r0,masterZ,Lambda_D,Lambda_E,MAP,n,k,t,nkt);
        auto t4 = std::chrono::high_resolution_clock::now();
        double err1 = (Lambda_E - E_prev).squaredNorm() / E_prev.squaredNorm();

        err = (err0 + err1 + err2)/3;
        all_err[n_itr] = err;
        n_itr++;


        if(timings)
        {
            double time_beta = std::chrono::duration<double, std::milli>(t2 - t1).count();
            double time_D    = std::chrono::duration<double, std::milli>(t3 - t2).count();
            double time_E    = std::chrono::duration<double, std::milli>(t4 - t3).count();
            
            Rcpp::Rcout << "--- Iteration " << n_itr << " Timings (ms) ---\n";
            Rcpp::Rcout << "estimate_beta2: " << time_beta << " ms\n";
            Rcpp::Rcout << "estimate_D:     " << time_D << " ms\n";
            Rcpp::Rcout << "estimate_E:     " << time_E << " ms\n";
        }

        if(verbose)
        {
            Rcpp::Rcout << "overall error after " << n_itr << " iterations: " << err << "\n";
            Rcpp::Rcout << "lambda D "  << printdims(Lambda_D) << err0 << " \n";
            Eigen::Index maxX, maxY;
            double max = (Lambda_D - D_prev).maxCoeff(&maxX,&maxY);
            Rcpp::Rcout <<  "Max: " << max << " (" << maxX << "," << maxY << ")\n";
            Rcpp::Rcout << Lambda_D(Eigen::seqN(maxX-3,6),Eigen::seqN(maxY-3,6)) << "\n\n";
            Rcpp::Rcout << D_prev(Eigen::seqN(maxX-3,6),Eigen::seqN(maxY-3,6)) << "\n"; 

            Rcpp::Rcout << "E " << err1 << " \n";
            Rcpp::Rcout << Lambda_E(Eigen::seqN(0,5)) << "\n\n";
            Rcpp::Rcout << E_prev(Eigen::seqN(0,5)) << "\n"; 

            Rcpp::Rcout << "beta " << err2 << " \n";
            Rcpp::Rcout << beta(Eigen::seqN(0,5)) << "\n\n";
            Rcpp::Rcout << beta_prev(Eigen::seqN(0,5)) << "\n"; 

            max = (beta - beta_prev).maxCoeff(&maxX,&maxY);
            Rcpp::Rcout <<  "Max: " << max << " (" << maxX << "," << maxY << ")\n";
            Rcpp::Rcout << beta(Eigen::seqN(maxX-3,6)) << "\n\n";

            Rcpp::Rcout << "sigma " << "\n" << sigma2 << "\n";
        }
    }
    Eigen::VectorXd E0 = Lambda_E.array().square();
    sigma2 = calc_sigma2(masterZ,D,E0,MAP,r0,n,k,t,p,REML);
    D = D * sigma2;
    Eigen::MatrixXd E = Eigen::MatrixXd::Zero(k,k);
    E.diagonal() = E0 * sigma2;

    bool converged = false;
    if(n_itr < max_itr) converged = true;

    Eigen::MatrixXd Et(k*t,k*t);
    Et_assemble_IP(E.diagonal(),Et,Eigen::MatrixXi::Constant(1,k*t,1),0,k,t,k*t);

    Eigen::MatrixXd ZDZ = masterZ * D * masterZ.transpose();
    Eigen::MatrixXd Sigma = ZDZ + Et;

    return(Rcpp::List::create(Rcpp::Named("Sigma")=Sigma,
           Rcpp::Named("E") = E,Rcpp::Named("D") = D, 
           Rcpp::Named("Lambda_E") = Lambda_E, 
           Rcpp::Named("beta") = beta, 
           Rcpp::Named("n_iter") = n_itr,
           Rcpp::Named("all_err") = all_err,
           Rcpp::Named("converged") = converged, 
           Rcpp::Named("sigma") = sigma2,
           Rcpp::Named("threshold")=theta));
}
