#define STRICT_R_HEADERS
#include "utils.h"

/*  
******************************************************************************
******************************************************************************
************ EVERY SUBJECT MUST HAVE ALL NODES AT AT LEAST 1 TIME ************
************                 OR AT LEAST A COLUMN                 ************
************         (ALL MATRICES NEED TO HAVE WIDTH 2K)         ************
******************************************************************************
******************************************************************************
*/

Eigen::VectorXd Zbcalc(const std::vector<Eigen::MatrixXd>& Z, 
                       const Eigen::VectorXd & b, int n, int nkt)
{
    Eigen::VectorXd Zb = Eigen::VectorXd::Zero(nkt);
    int cnt = 0;
    for(int i = 0; i < n; ++i)
    {
        int kt = Z[i].rows();
        Zb(Eigen::seqN(cnt,kt)) = Z[i] * b; 
        cnt += kt;
    }
    return(Zb);
}

Eigen::VectorXd Zrcalc(const std::vector<Eigen::MatrixXd>& Z, 
                       const Eigen::VectorXd & r, int n)
{
    int kt = Z[0].rows();
    Eigen::VectorXd Zb = Z[0].transpose() * r(Eigen::seqN(0,kt));
    int cnt = kt;
    for(int i = 1; i < n; ++i)
    {
        kt = Z[i].rows();
        Zb += Z[i].transpose() * r(Eigen::seqN(cnt,kt)); 
        cnt += kt;
    }
    return(Zb);
}


void calc_b(const Eigen::MatrixXd & X, const Eigen::VectorXd & r0, 
            const std::vector<Eigen::MatrixXd> & Z,
            const Eigen::MatrixXd & Lambda_D, const Eigen::VectorXd & E, Eigen::VectorXd & b,
            const Eigen::MatrixXi & MAP,
            int n, int k, int t, int nkt)
{
    int q = 2*k;
    Eigen::MatrixXd ZEEZ = Eigen::MatrixXd::Zero(q,q);
    Eigen::MatrixXd DZETE = Eigen::MatrixXd::Zero(q,nkt);
    Eigen::VectorXd EInv = Eigen::ArrayXd::Constant(k,1.0) / E.array(); //E.inverse();
    int cnt = 0;
    for(int i=0;i<n;++i)
    {
        int kt = Z[i].rows();
        //Eigen::MatrixXd Et = Et_assemble(E, MAP, i, k, t, kt);
        Eigen::MatrixXd EtInv = Et_assemble(EInv, MAP, i, k, t, kt);
        Eigen::MatrixXd EZ = EtInv * Z[i]; //Z(Eigen::seqN(i*k*t,k*t),Eigen::all); 
        ZEEZ += EZ.transpose() * EZ;
        DZETE(Eigen::all,Eigen::seqN(cnt,kt)) = Lambda_D.transpose() * Z[i].transpose() * EtInv.array().pow(2).matrix(); //Et.array().pow(2).matrix().inverse();
        cnt += kt;
    }

    b = Lambda_D.transpose() * (Eigen::MatrixXd::Identity(2*k,2*k) + (Lambda_D.transpose() * ZEEZ * Lambda_D)).colPivHouseholderQr().solve(DZETE) * r0;
}

void estimate_E(const Eigen::MatrixXd & X, const Eigen::VectorXd & r0, 
                const std::vector<Eigen::MatrixXd>& Z,
                const Eigen::MatrixXd & Lambda_D, Eigen::VectorXd & Lambda_E,
                const Eigen::MatrixXi & MAP, 
                int n, int k, int t, int nkt)
{
    Eigen::VectorXd b;
    calc_b(X,r0,Z,Lambda_D,Lambda_E,b,MAP,n,k,t,nkt);
    Eigen::VectorXd r = r0 - Zbcalc(Z, b, n, nkt);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n*t,k);
    int current = 0;
    for(int i=0; i<n; ++i)
    {
        for(int j=0; j<k; ++j)
        {
            for(int l=0; l<t; ++l)
            {
                if (MAP(i,j*t+l) == 1)
                {
                    R(i*t + l,j) = r(current);
                    current++;
                }
            }
        }
    }
    Lambda_E = covCalc(R,MAP).diagonal().array().sqrt();
}

void calc_e(const Eigen::VectorXd & r0, const std::vector<Eigen::MatrixXd>& Z,
            const Eigen::VectorXd & E, const Eigen::MatrixXd & Lambda_D, 
            const Eigen::MatrixXi & MAP, Eigen::VectorXd & e, 
            int n, int k, int t, int nkt)
{
    int p = 2*k;
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(p,p);
    for(int i=0;i<n;++i)
    {
        B += Z[i].transpose() * Z[i];
    }
    Eigen::MatrixXd BDBinv = B * Lambda_D * Lambda_D.transpose() * B.transpose();
    Eigen::MatrixXd BD = Lambda_D.colPivHouseholderQr().solve(B.inverse());
    Eigen::MatrixXd BDB = BD.transpose() * BD;
    Eigen::MatrixXd ZEEZ = Eigen::MatrixXd::Zero(p,p);
    Eigen::MatrixXd EZ = Eigen::MatrixXd::Zero(nkt,p);
    Eigen::MatrixXd EZ_tmp;

    int cnt = 0;
    for(int i=0;i<n;++i)
    {
        int kt = Z[i].rows();
        EZ_tmp = Et_assemble(E, MAP, i, k, t, kt) * Z[i];
        EZ(Eigen::seqN(cnt,kt),Eigen::all) = EZ_tmp;
        ZEEZ += EZ_tmp.transpose() * EZ_tmp;
        cnt += kt;
    }
    //H = (B.transpose() * Lambda_D.transpose() * Lambda_D * B + ZEEZ).transpose().llt().solve(H.transpose()).transpose();
    Eigen::MatrixXd H = EZ * (BDBinv + ZEEZ).inverse();
    Eigen::VectorXd Zr = Zrcalc(Z,r0,n);
    
    cnt = 0; 
    for(int i=0;i<n;++i)
    {
        int kt = Z[i].rows();
        Eigen::MatrixXd Et = Et_assemble(E, MAP, i, k, t, kt);
        e(Eigen::seqN(cnt,kt)) = Et * (Et * Z[i] - H(Eigen::seqN(cnt,kt),Eigen::all) * ZEEZ) * BDB * Zr;
        cnt += kt;
    }
}


void a2_thresholdRange(const Eigen::MatrixXd & R, Eigen::ArrayXXd& theta, Eigen::MatrixXd& cov, 
                    const Eigen::MatrixXi & MAP, double & lower, double & upper)
{
    int n = R.rows();
    int p = R.cols();
    cov = covCalc(R,MAP);

    theta = (RtR(R,MAP) - cov.array().pow(2).matrix()).array().sqrt();
    Eigen::MatrixXd delta = (cov.array() / theta).cwiseAbs().matrix();
    delta.diagonal() = Eigen::VectorXd::Zero(delta.rows());
    upper = delta.maxCoeff();
    lower = (delta.array() <= 0.f).select(std::numeric_limits<int>::max(), delta).minCoeff();
}

void a2_threshold(const Eigen::MatrixXd& abscov, const Eigen::MatrixXd& signcov, 
               const Eigen::MatrixXd& thetalambda, Eigen::MatrixXd& sigma_out)
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

void a2_threshold_D(const Eigen::MatrixXd & R, Eigen::MatrixXd& sigma, 
                 Eigen::ArrayXXd & theta, const Eigen::MatrixXi & MAP, 
                 int n_fold=5)
{
    int n = R.rows();
    int p = R.cols();
    int nParam = 100;
    Eigen::MatrixXd cov(p,p);
    double lower = 0.0;
    double upper = 0.0;
    
    a2_thresholdRange(R,theta,cov,MAP,lower,upper);
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
    for (int i=0;i<n_fold; ++i)
    {
        std::deque<int> val_idx;
        std::deque<int> not_val_idx;
        find_all(part,i,val_idx,not_val_idx);
        Eigen::MatrixXd covTest = covCalc(R(val_idx,Eigen::all),MAP(val_idx,Eigen::all));
        double lower = 0.0;
        double upper = 0.0;
        Eigen::MatrixXd absCovTrain(p,p);
        Eigen::ArrayXXd thetaTrain(p,p);
        a2_thresholdRange(R(not_val_idx,Eigen::all), thetaTrain, absCovTrain, MAP(not_val_idx,Eigen::all), lower, upper);
        Eigen::ArrayXXd signCovTrain = absCovTrain.cwiseSign();
        absCovTrain = absCovTrain.cwiseAbs();
        for(int j=0;j<nParam;++j)
        {
            //Eigen::MatrixXd thetalambda = params[j] * thetaTrain;
            Eigen::MatrixXd sigmaTrain(p,p);
            a2_threshold(absCovTrain,signCovTrain,params[j] * thetaTrain,sigmaTrain);
            error(i,j) = (sigmaTrain - covTest).norm();
        }
    }
    Eigen::Index minIndex;
    error.colwise().sum().minCoeff(&minIndex);
    Eigen::ArrayXXd absCov = cov.cwiseAbs();
    Eigen::ArrayXXd signCov = cov.cwiseSign();

    a2_threshold(absCov,signCov,params[minIndex] * theta,sigma);

}

void estimate_D(const Eigen::MatrixXd & X, const Eigen::VectorXd & r0, 
                const std::vector<Eigen::MatrixXd>& Z,
                const Eigen::VectorXd & E, Eigen::MatrixXd & Lambda_D, 
                const Eigen::MatrixXi & MAP, Eigen::MatrixXd & D,
                int n, int k, int t, int nkt, bool soft=1)
{
    int p = 2*k;
    Eigen::VectorXd e = Eigen::VectorXd::Zero(nkt);
    calc_e(r0,Z,E,Lambda_D,MAP,e,n,k,t,nkt);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n,p);
    Eigen::VectorXd r = r0 - e;
    int cnt = 0;
    for(int i=0;i<n;++i)
    {
        int kt = Z[i].rows();
        R(i,Eigen::all) = (Z[i].transpose() * Z[i]).colPivHouseholderQr().solve(Z[i].transpose()) * r(Eigen::seqN(cnt,kt)); 
        cnt += kt;
    }

    Eigen::ArrayXXd theta(2*k,2*k);
    a2_threshold_D(R,D,theta,MAP);
    Lambda_D = (D + Eigen::MatrixXd::Identity(p,p)).llt().matrixL();

}

double calc_sigma2(const std::vector<Eigen::MatrixXd>& Z, 
                   const Eigen::MatrixXd& D, const Eigen::VectorXd& E, 
                   const Eigen::MatrixXi & MAP,
                   const Eigen::VectorXd& r0, 
                   int n, int k, int t, int p, 
                   bool REML=false)
{
    double sigma2 = 0.0;
    Eigen::MatrixXd Lambda_V;
    int nkt = 0;
    for(int i=0; i<n;++i)
    {
        int kt = Z[i].rows();
        Eigen::MatrixXd Et = Et_assemble(E, MAP, i, k, t, kt);
        Lambda_V = (Z[i] * D * Z[i].transpose()  + Et).llt().matrixL();
        sigma2 += (Lambda_V.colPivHouseholderQr().solve(r0(Eigen::seqN(nkt,kt)))).squaredNorm();
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

void a2_initial_estimates(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, 
                       const std::vector<Eigen::MatrixXd>& Z, 
                       const Eigen::MatrixXi & MAP, 
                       std::vector<Eigen::MatrixXd> & Sigma_list, 
                       Eigen::MatrixXd & Lambda_D, Eigen::MatrixXd & D, 
                       Eigen::VectorXd & Lambda_E, Eigen::VectorXd & beta, 
                       Eigen::VectorXd & r,
                       int n, int k, int t)
{
    int p = X.cols();
    int q = 2*k; //Z.cols();
    beta = (X.transpose() * X).ldlt().solve(X.transpose()) * y;
    r = y - X * beta;
    Eigen::MatrixXd R(n,k*t);
    int nkt = 0;
    for(int i = 0; i<n; ++i)
    {
        int kt0 = MAP.rowwise().sum()(i);
        std::deque<int> idxs;
        std::deque<int> waste;
        find_all(MAP(i,Eigen::all),1,idxs,waste);
        R(i,idxs) = r(Eigen::seqN(nkt,kt0));
        nkt += kt0;
    }
    Eigen::MatrixXd cov_int = covCalc(R, MAP);
    for(int i=0;i<k;++i)
    {
        Lambda_E(i) = (cov_int.diagonal().array())(Eigen::seqN(i*t,t)).sqrt().mean();
    }

    cov_int.diagonal() = cov_int.diagonal();

    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(q,q); 
    Eigen::MatrixXd ZCZ = Eigen::MatrixXd::Zero(q,q);
    for(int i=0;i<n;++i)
    {
        B += Z[i].transpose() * Z[i];
        ZCZ += Z[i].transpose() * cov_int * Z[i];
    }
    D = B.colPivHouseholderQr().solve(ZCZ) * B.inverse();
    Lambda_D = (D + Eigen::MatrixXd::Identity(q,q)).llt().matrixL();
    calc_ZDZ_plus_E_list(Z,D,Lambda_E.array().pow(2),Sigma_list,MAP,n,k,t);
}


// [[Rcpp::export]]
Rcpp::List estimate_DEbeta(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, 
                           Rcpp::List & Z_in, //Z_in will be destroyed for space sacing reasons
                           int n, int k, int t,
                           int max_itr=250, 
                           double convergence_cutoff=0.0001,
                           bool REML=false,
                           bool verbose=false,
                           int seed=1234)
{
    int p = X.cols();
    std::vector<Eigen::MatrixXd> Sigma_list(n);
    std::vector<Eigen::MatrixXd> Z(n);
    list2vec(Z,Z_in);
    Z_in = Rcpp::List();
    Eigen::MatrixXd Lambda_D(2*k,2*k);
    Eigen::MatrixXd D(2*k,2*k);
    Eigen::VectorXd Lambda_E(k);
    Eigen::VectorXd beta(p);
    Eigen::VectorXd r0;

    Eigen::MatrixXi MAP = Eigen::MatrixXi::Zero(n,k*t);
    Eigen::MatrixXd masterZ(k*t,2*k);
    int nkt = make_MAP(Z,masterZ,MAP,r0,n,k,t);
    
    a2_initial_estimates(X,y,Z,MAP,Sigma_list,Lambda_D,D,Lambda_E,beta,r0,n,k,t);

    Eigen::VectorXi kt_vec = MAP.rowwise().sum();
    Eigen::VectorXd beta_offset = Eigen::VectorXd::Zero(p);
    
    double sigma2 = 0.0;
    double err = 10.0;
    double prev_err = 9.0;
    double double_prev_err = 8.0;
    int n_itr = 0;
    std::vector<double> all_err(max_itr);
    while (((err > convergence_cutoff) || (prev_err > convergence_cutoff)) && (n_itr < max_itr))
    {
        double_prev_err = prev_err;
        prev_err = err;
        Eigen::MatrixXd D_prev = Lambda_D;
        Eigen::VectorXd E_prev = Lambda_E;
        Eigen::VectorXd beta_prev = beta;


        estimate_beta(X,y,kt_vec,MAP,Sigma_list,beta,n,k,t, n_itr);
        beta += beta_offset;
        r0 = y - X * beta; 
        estimate_D(X,r0,Z,Lambda_E.array().pow(2),Lambda_D,MAP,D,n,k,t,nkt);
        estimate_E(X,r0,Z,Lambda_D,Lambda_E,MAP,n,k,t,nkt);

        // TODO: delete? 
        sigma2 = calc_sigma2(Z,D,Lambda_E.array().pow(2),MAP,r0,n,k,t,p,REML);
        //Lambda_D = Lambda_D * std::sqrt(sigma2);
        //Lambda_E = Lambda_E * std::sqrt(sigma2);
        //D = D * sigma2;

        calc_ZDZ_plus_E_list(Z,D * sigma2,Lambda_E.array().pow(2) * sigma2, Sigma_list, MAP, n, k, t);

        double err0 = (Lambda_D - D_prev).squaredNorm() / (k*(2*k+1));
        double err1 = (Lambda_E - E_prev).squaredNorm() / k;
        double err2 = pow((beta - beta_prev).sum(),2) / p;

        err = (err0 + err1 + err2)/3;
        all_err[n_itr] = err;
        n_itr++;

        //TODO: keep checking if this works
        beta_offset = Eigen::VectorXd::Zero(p);

        // If the last three error values are all the same we are stuck in a loop
        // add a random offset to beta
        if((double_prev_err - prev_err) < convergence_cutoff && 
           (prev_err - err) < convergence_cutoff && 
           (n_itr < max_itr))
        {
            std::random_device rd{};
            std::mt19937 gen{rd()};
            double mean = 0;
            double var = (beta.array() - beta.mean()).square().sum() / 1000.0;
            std::normal_distribution<double> distribution(mean, var);

            // Draw a sample from the normal distribution
            Rcpp::Rcout << "offsetting betas \n";
            for(int i; i < p; ++i)
            {
                beta_offset(i) = distribution(gen);
            }
        }

        if(verbose)
        {
            Rcpp::Rcout << "overall error after " << n_itr << " iterations: " << err << "\n";
            Rcpp::Rcout << "D " << err0 << " \n";
            Rcpp::Rcout << Lambda_D(Eigen::seqN(0,5),Eigen::seqN(0,5)) << "\n\n";
            Rcpp::Rcout << D_prev(Eigen::seqN(0,5),Eigen::seqN(0,5)) << "\n"; 

            Rcpp::Rcout << "E " << err1 << " \n";
            Rcpp::Rcout << Lambda_E(Eigen::seqN(0,5)) << "\n\n";
            Rcpp::Rcout << E_prev(Eigen::seqN(0,5)) << "\n"; 

            Rcpp::Rcout << "beta " << err2 << " \n";
            Rcpp::Rcout << beta(Eigen::seqN(0,5)) << "\n\n";
            Rcpp::Rcout << beta_prev(Eigen::seqN(0,5)) << "\n"; 

            Rcpp::Rcout << "sigma " << "\n" << sigma2 << "\n";
        }
    }
    Eigen::VectorXd E0 = Lambda_E.array().pow(2);
    sigma2 = calc_sigma2(Z,D,E0,MAP,r0,n,k,t,p,REML);
    D = D * sigma2;
    Eigen::MatrixXd E = Eigen::MatrixXd::Zero(k,k);
    E.diagonal() = E0 * sigma2;

    bool converged = false;
    if(n_itr < max_itr) converged = true;

    Eigen::MatrixXd Et = Et_assemble(E.diagonal(),Eigen::MatrixXi::Constant(1,k*t,1),0,k,t,k*t);

    return(Rcpp::List::create( Rcpp::Named("Sigma")=masterZ * D * masterZ.transpose() + Et,//Rcpp::Named("Sigma")=Rcpp::wrap(Sigma_list),
           Rcpp::Named("E") = E,Rcpp::Named("D") = D, 
           Rcpp::Named("Lambda_E") = Lambda_E, 
           Rcpp::Named("beta") = beta, 
           Rcpp::Named("n_iter") = n_itr,
           Rcpp::Named("all_err") = all_err,
           Rcpp::Named("converged") = converged, 
           Rcpp::Named("sigma") = sigma2,
           Rcpp::Named("MAP")= MAP));
}
