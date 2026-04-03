#define STRICT_R_HEADERS
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
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

void build_V_list_from_master(std::vector<Eigen::MatrixXd> & V, const Eigen::MatrixXd & masterV, const Eigen::MatrixXi & MAP, int n, int k, int t)
{
    Eigen::MatrixXi kt_vec = MAP.rowwise().sum();
    for(int i=0; i<n; ++i)
    {
        int kt = kt_vec(i);
        Eigen::MatrixXd Vi(kt,kt);
        V_assemble_IP(masterV,Vi,MAP,i,k,t,kt);
        V[i] = Vi;
    }
}

void V_assemble_IP(const Eigen::Ref<const Eigen::MatrixXd> & masterV, 
                   Eigen::MatrixXd & V_out,
                   const Eigen::Ref<const Eigen::MatrixXi> & MAP,
                   int i, int k, int t, int kt)
{
    V_out.setZero(kt,kt);
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
                    V_out(cnt0,cnt1) = masterV(j,l);
                    cnt1++;
                }
            }
            cnt0++;
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
Eigen::MatrixXd covCalc(const Eigen::MatrixXd & X, const Eigen::MatrixXi & MAP)
{
    int n = X.rows();
    int p = X.cols();
    int full_kt = MAP.cols();

    Eigen::MatrixXd SumXY = Eigen::MatrixXd::Zero(p, p);
    Eigen::MatrixXd N = Eigen::MatrixXd::Zero(p, p);
    Eigen::MatrixXd SumX_shared = Eigen::MatrixXd::Zero(p, p);

    for(int i = 0; i < n; ++i)
    {
        // 1. Collect non-zero indices to skip the vast majority of the empty matrix
        std::vector<int> active;
        active.reserve(p);
        // DYNAMIC MASKING: Prevent Out-of-Bounds Memory Reads
        if (p == full_kt) {
            // Phase 1: Residual Covariance (X is n x kt)
            for(int c = 0; c < p; ++c) {
                if(MAP(i, c) != 0) active.push_back(c);
            }
        } else {
            // Phase 2: Random Effects Covariance (X is n x 2k)
            int timepts = full_kt / (p / 2);
            for(int c = 0; c < p; ++c) {
                int node = c / 2; // Map slope/intercept back to the node
                bool has_data = false;
                for(int time = 0; time < timepts; ++time) {
                    if(MAP(i, node * timepts + time) != 0) {
                        has_data = true; break;
                    }
                }
                if(has_data) active.push_back(c);
            }
        }

        // 2. Only loop over active observation pairs!
        for(size_t idx1 = 0; idx1 < active.size(); ++idx1)
        {
            int c1 = active[idx1];
            double x1 = X(i, c1);
            for(size_t idx2 = 0; idx2 <= idx1; ++idx2)
            {
                int c2 = active[idx2];
                double x2 = X(i, c2);

                SumXY(c1, c2) += x1 * x2;
                N(c1, c2) += 1.0;
                SumX_shared(c1, c2) += x1;
                if(c1 != c2) {
                    SumX_shared(c2, c1) += x2;
                }
            }
        }
    }

    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(p,p);
    for(int i=0; i < p; ++i)
    {
        for(int j=0; j <= i; ++j)
        {
            double Nij = N(i,j);
            if(Nij > 1.0)
            {
                double val = (SumXY(i, j) - (SumX_shared(i, j) * SumX_shared(j, i) / Nij)) / Nij;
                cov(i,j) = val;
                if(i != j) cov(j,i) = val;
            }
        }
    }
    return cov;
}

void get_cov_stats(const Eigen::Ref<const Eigen::MatrixXd>& R, 
                   const Eigen::Ref<const Eigen::MatrixXi>& MAP, 
                   const std::vector<int>& indices,
                   Eigen::MatrixXd& SumXY, Eigen::MatrixXd& N, 
                   Eigen::MatrixXd& SumX_shared, Eigen::MatrixXd& SumRsq)
{
    int p = R.cols();
    int full_kt = MAP.cols();

    SumXY.setZero(p, p);
    N.setZero(p, p);
    SumX_shared.setZero(p, p);
    SumRsq.setZero(p, p);

    for(int row_idx : indices) 
    {
        // 1. Find active nodes for this subject
        std::vector<int> active;
        active.reserve(p);
        // SAFE MAPPING
        if (p == full_kt) {
            for(int c = 0; c < p; ++c) {
                if(MAP(row_idx, c) != 0) active.push_back(c);
            }
        } else {
            int timepts = full_kt / (p / 2);
            for(int c = 0; c < p; ++c) {
                int node = c / 2;
                bool has_data = false;
                for(int time = 0; time < timepts; ++time) {
                    if(MAP(row_idx, node * timepts + time) != 0) {
                        has_data = true; break;
                    }
                }
                if(has_data) active.push_back(c);
            }
        }

        // 2. Accumulate raw sums for the active pairs
        for(size_t idx1 = 0; idx1 < active.size(); ++idx1) 
        {
            int c1 = active[idx1];
            double r1 = R(row_idx, c1);
            double rsq1 = r1 * r1;
            
            for(size_t idx2 = 0; idx2 <= idx1; ++idx2) 
            {
                int c2 = active[idx2];
                double r2 = R(row_idx, c2);
                double rsq2 = r2 * r2;

                SumXY(c1, c2) += r1 * r2;
                N(c1, c2) += 1.0;
                SumX_shared(c1, c2) += r1;
                SumRsq(c1, c2) += rsq1 * rsq2;
                
                if(c1 != c2) {
                    SumX_shared(c2, c1) += r2;
                }
            }
        }
    }
}

void build_cov_and_theta(const Eigen::MatrixXd& SumXY, const Eigen::MatrixXd& N, 
                         const Eigen::MatrixXd& SumX_shared, const Eigen::MatrixXd& SumRsq,
                         Eigen::MatrixXd& cov, Eigen::ArrayXXd& theta)
{
    int p = SumXY.cols();
    cov = Eigen::MatrixXd::Zero(p, p);
    theta = Eigen::ArrayXXd::Zero(p, p);

    for(int i = 0; i < p; ++i) 
    {
        for(int j = 0; j <= i; ++j) 
        {
            double Nij = N(i, j);
            if(Nij > 1.0) 
            {
                // 1. Covariance Formula
                double val = (SumXY(i, j) - (SumX_shared(i, j) * SumX_shared(j, i) / Nij)) / Nij;
                cov(i, j) = val;
                
                // 2. R^T R Formula
                double rtr_val = SumRsq(i, j) / Nij;
                
                // 3. Theta (Variance) Formula -> max(0.0) protects against microscopic float noise
                double theta_val = std::max(0.0, rtr_val - (val * val));
                theta(i, j) = std::sqrt(theta_val);
                
                // Mirror to upper triangle
                if(i != j) {
                    cov(j, i) = val;
                    theta(j, i) = theta(i, j);
                }
            }
        }
    }
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

void find_all(const std::vector<int> & vec, const int & val, std::vector<int> & out_val, std::vector<int> & out_not_val)
{
    out_val.reserve(vec.size());
    out_not_val.reserve(vec.size());
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

void find_all(const Eigen::VectorXi & vec, const int & val, std::vector<int> & out_val, std::vector<int> & out_not_val)
{
    out_val.reserve(vec.size());
    out_not_val.reserve(vec.size());
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
            Et.diagonal()(cnt) = E(j);
            cnt++;
        }
        cnt2++;
        }
    }
  return(Et);
}

//[[Rcpp::export]]
Eigen::MatrixXd Z_assemble(const Eigen::MatrixXd & masterZ, 
                           const Eigen::MatrixXi & MAP,
                           int i, int k, int t, int kt)
{
    Eigen::MatrixXd Z_out = Eigen::MatrixXd::Zero(kt,2*k);
    int cnt = 0;
    for(int j = 0; j<k*t; ++j)
    {
        if(MAP(i,j) == 1)
        {
            Z_out.row(cnt).noalias() = masterZ.row(j);
            cnt++;
        }
    }
    return(Z_out);
}

void Et_assemble_IP(const Eigen::Ref<const Eigen::VectorXd> & E, 
                    Eigen::VectorXd & Et,
                    const Eigen::Ref<const Eigen::MatrixXi> & MAP, 
                    int i, int k, int t, int kt)
{
    // Function body remains identical
    Et.resize(kt);
    int cnt = 0;
    int cnt2 = 0;
    for(int j = 0; j < k; ++j)
    {
        for(int l = 0; l < t; ++l)
        {
        if(MAP(i,cnt2) == 1)
        {
            Et(cnt) = E(j);
            cnt++;
        }
        cnt2++;
        }
    }
}

void Z_assemble_IP(const Eigen::Ref<const Eigen::MatrixXd> & masterZt, 
                   Eigen::MatrixXd & Zt_out,
                   const Eigen::Ref<const Eigen::MatrixXi> & MAP,
                   int i, int k, int t, int kt)
{
    // Function body remains identical
    Zt_out.resize(2*k,kt);
    int cnt = 0;
    for(int j = 0; j<k*t; ++j)
    {
        if(MAP(i,j) == 1)
        {
            Zt_out.col(cnt).noalias() = masterZt.col(j);
            cnt++;
        }
    }
}

void Et_Z_assemble_IP(const Eigen::Ref<const Eigen::VectorXd> & masterE, 
                   Eigen::VectorXd & Et_out,
                   const Eigen::Ref<const Eigen::MatrixXd> & masterZt, 
                   Eigen::MatrixXd & Zt_out,
                   const Eigen::Ref<const Eigen::MatrixXi> & MAP,
                   int i, int k, int t, int kt)
{
    Zt_out.resize(2*k,kt);
    Et_out.resize(kt);
    int cnt = 0;
    int cnt2 = 0;
    for(int j = 0; j<k; ++j)
    {
        for(int l = 0; l < t; ++l)
        {
            if(MAP(i,cnt2) == 1)
            {
                Et_out(cnt) = masterE(j);
                Zt_out.col(cnt).noalias() = masterZt.col(j);
                cnt++;
            }
        }
        cnt2++;
    }
}

void calc_ZDZ_plus_E_list(const Eigen::MatrixXd & masterZt,
                          const Eigen::MatrixXd & D, const Eigen::VectorXd & E,
                          std::vector<Eigen::MatrixXd> & out, 
                          const Eigen::MatrixXi & MAP,
                          int n, int k, int t)
{
    Eigen::MatrixXi kt_vec = MAP.rowwise().sum();
    Eigen::MatrixXd Zti(2*k,k*t), Vi(k*t,k*t);
    Eigen::VectorXd Et(k*t);
    for(int i=0; i<n; ++i)
    {
        int kt = kt_vec(i);
        if(kt == 0) continue;
        Z_assemble_IP(masterZt,Zti,MAP,i,k,t,kt);
        Et_assemble_IP(E,Et,MAP,i,k,t,kt);
        Vi = Zti.transpose() * D * Zti;
        Vi.diagonal() += Et;
        out[i] = Vi;
    }
}

// [[Rcpp::export]]
Rcpp::List calc_ZDZ_plus_E_list(const Eigen::MatrixXd & masterZt,
                          const Eigen::MatrixXd & D, const Eigen::VectorXd & E,
                          const Eigen::MatrixXi & MAP,
                          int n, int k, int t)
{
    Eigen::VectorXd r0;
    std::vector<Eigen::MatrixXd> out(n);
    calc_ZDZ_plus_E_list(masterZt,D,E,out,MAP,n,k,t);
    return(Rcpp::wrap(out));
}

void calc_ZDZ_plus_E(const Eigen::MatrixXd & masterZt,
                    const Eigen::MatrixXd & D, 
                    const Eigen::VectorXd & E,
                    Eigen::MatrixXd & out, 
                    const Eigen::MatrixXi & MAP,
                    int n, int k, int t, int nkt)
{
    out.resize(nkt,nkt);
    Eigen::VectorXi kt_vec = MAP.rowwise().sum();
    Eigen::MatrixXd Zti(k*t,2*k), Vi(k*t,k*t);
    Eigen::VectorXd Et(k*t);
    int cnt=0;
    for(int i=0; i<n; ++i)
    {
        int kt = kt_vec(i);
        if(kt == 0) continue;
        Z_assemble_IP(masterZt,Zti,MAP,i,k,t,kt);
        Et_assemble_IP(E,Et,MAP,i,k,t,kt);
        Vi = Zti.transpose() * D * Zti;
        Vi.diagonal() += Et;
        out.block(cnt,cnt,kt,kt) =Vi;
        cnt += kt;
    }
}

// [[Rcpp::export]]
Rcpp::List calc_ZDZ_plus_E(const Eigen::MatrixXd & masterZt,
                          const Eigen::MatrixXd & D,
                          const Eigen::VectorXd & E,
                          const Eigen::MatrixXi & MAP,
                          int n, int k, int t, int nkt)
{
    Eigen::VectorXd r0;
    Eigen::MatrixXd masterZ;
    Eigen::MatrixXd out(nkt,nkt);
    calc_ZDZ_plus_E(masterZt,D,E,out,MAP,n,k,t,nkt);
    return(Rcpp::wrap(out));
}


void estimate_beta(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, 
                   const Eigen::VectorXi kt_vec, const Eigen::MatrixXi & MAP,
                   const Eigen::MatrixXd & masterV, Eigen::VectorXd & beta,
                   int n, int k, int t, bool verbose, double eigen_threshold)
{
    int q = X.cols();
    Eigen::MatrixXd XVX = Eigen::MatrixXd::Zero(q,q);
    Eigen::VectorXd XVy = Eigen::VectorXd::Zero(q);
    Eigen::MatrixXd Vi(k*t,k*t), Xi;
    Eigen::VectorXd yi;

    int cnt = 0;
    for(int i=0; i<n; ++i)
    {
        int kt = kt_vec(i);
        if(kt == 0) continue;
        V_assemble_IP(masterV,Vi,MAP,i,k,t,kt);

        // Map the current subject's X and y
        Xi = X.block(cnt, 0, kt, q);
        yi = y.segment(cnt, kt);

        Eigen::LLT<Eigen::MatrixXd> llt_Vi(Vi);
        if(llt_Vi.info() == Eigen::Success)
        {
            XVX += Xi.transpose() * llt_Vi.solve(Xi);
            XVy += Xi.transpose() * llt_Vi.solve(yi);
        }
        else{ // fallback if Vi not invertible
            Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod_Vi;
            cod_Vi.setThreshold(eigen_threshold);
            
            cod_Vi.compute(Vi);
            //Eigen::MatrixXd Vi_inv = cod_Vi.pseudoInverse();
            // Reconstruct the safely inverted matrix
            //Eigen::MatrixXd Vi_inv = evecs * evals.asDiagonal() * evecs.transpose();

            XVX += Xi.transpose() * cod_Vi.solve(Xi);//Vi_inv * Xi;
            XVy += Xi.transpose() * cod_Vi.solve(yi);//Vi_inv * yi;
        }
        cnt += kt;
    }
    Eigen::LDLT<Eigen::MatrixXd> ldlt_XVX(XVX);
    if(ldlt_XVX.info() == Eigen::Success)
    {
        beta = ldlt_XVX.solve(XVy);
    }
    else
    {
        if(verbose) {
            Rcpp::Rcout << "Warning: XVX matrix is not positive definite. Using pseudo-inverse fallback for beta estimation.\n";
        }
        beta = XVX.completeOrthogonalDecomposition().pseudoInverse() * XVy;
    }
    if(verbose)
    {
        Rcpp::Rcout << "XVX" << printdims(XVX) << std::endl;
        Rcpp::Rcout << XVX.block(0,0,5,5) << std::endl;
    }

}

void estimate_beta2(const Eigen::Ref<const Eigen::MatrixXd> & X, 
                    const Eigen::Ref<const Eigen::VectorXd> & y, 
                    const Eigen::Ref<const Eigen::MatrixXd> & Zt,
                    const Eigen::Ref<const Eigen::MatrixXd> & D,
                    const Eigen::Ref<const Eigen::VectorXd> & E,
                    const Eigen::VectorXi & kt_vec, 
                    const Eigen::Ref<const Eigen::MatrixXi> & MAP,
                    Eigen::VectorXd & beta,
                    int n, int k, int t)
{
    int q = X.cols();
    Eigen::MatrixXd XVX = Eigen::MatrixXd::Zero(q,q);
    Eigen::VectorXd XVy = Eigen::VectorXd::Zero(q);
    // 1. Invert D EXACTLY ONCE outside the loop (Massive CPU saving!)
    Eigen::MatrixXd D_inv;
    Eigen::LDLT<Eigen::MatrixXd> ldlt_D(D);
    if(ldlt_D.info() == Eigen::Success) {
        D_inv = ldlt_D.solve(Eigen::MatrixXd::Identity(2*k, 2*k));
    } else {
        D_inv = D.completeOrthogonalDecomposition().pseudoInverse();
    }

    Eigen::MatrixXd Zti, Xi, ZiX, M;
    Eigen::VectorXd yi, E_inv, Ziy;
    int cnt = 0;
    
    for(int i = 0; i < n; ++i)
    {
        int kt = kt_vec(i);
        
        // Get Z for this subject
        Z_assemble_IP(Zt, Zti, MAP, i, k, t, kt);
        
        // Bypass creating a dense Et matrix. Just grab the inverted diagonal
        E_inv.resize(kt);
        int cnt2 = 0, c = 0;
        for(int j = 0; j < k; ++j) {
            for(int l = 0; l < t; ++l) {
                if(MAP(i, cnt2) == 1) {
                    E_inv(c++) = 1.0 / E(j); 
                }
                cnt2++;
            }
        }
        
        // Map the current subject's X and y
        Xi = X.block(cnt, 0, kt, q);
        yi = y.segment(cnt, kt);
        
        // The Woodbury Transformation Variables
        Eigen::MatrixXd X_tilde = E_inv.asDiagonal() * Xi;
        Eigen::VectorXd y_tilde = E_inv.asDiagonal() * yi;
        
        // Accumulate the base E^-1 terms
        XVX.noalias() += Xi.transpose() * X_tilde;
        XVy.noalias() += Xi.transpose() * y_tilde;
        
        // Inner Woodbury components
        ZiX = Zti * X_tilde;  
        Ziy = Zti * y_tilde;  
        
        // Inner matrix M = D^-1 + Z^T * E^-1 * Z  (Always exactly 2k x 2k)
        M = D_inv + Zti * E_inv.asDiagonal() * Zti.transpose(); 
        
        // Fast 2k x 2k decomposition
        Eigen::LDLT<Eigen::MatrixXd> ldlt_M(M);
        if(ldlt_M.info() == Eigen::Success)
        {
            XVX.noalias() -= ZiX.transpose() * ldlt_M.solve(ZiX);
            XVy.noalias() -= ZiX.transpose() * ldlt_M.solve(Ziy);
        }
        else
        {
            Eigen::MatrixXd M_inv = M.completeOrthogonalDecomposition().pseudoInverse();
            XVX.noalias() -= ZiX.transpose() * M_inv * ZiX;
            XVy.noalias() -= ZiX.transpose() * M_inv * Ziy;
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

Eigen::MatrixXd RtR(const Eigen::MatrixXd & R, const Eigen::MatrixXi & MAP)
{
    int n = R.rows();
    int p = R.cols();
    int full_kt = MAP.cols();

    Eigen::MatrixXd SumRsq = Eigen::MatrixXd::Zero(p, p);
    Eigen::MatrixXd N = Eigen::MatrixXd::Zero(p, p);

    for(int i = 0; i < n; ++i)
    {
        std::vector<int> active;
        active.reserve(p);
        // SAFE MAPPING
        if (p == full_kt) {
            for(int c = 0; c < p; ++c) {
                if(MAP(i, c) != 0) active.push_back(c);
            }
        } else {
            int timepts = full_kt / (p / 2);
            for(int c = 0; c < p; ++c) {
                int node = c / 2;
                bool has_data = false;
                for(int time = 0; time < timepts; ++time) {
                    if(MAP(i, node * timepts + time) != 0) {
                        has_data = true; break;
                    }
                }
                if(has_data) active.push_back(c);
            }
        }

        for(size_t idx1 = 0; idx1 < active.size(); ++idx1)
        {
            int c1 = active[idx1];
            double rsq1 = R(i, c1) * R(i, c1); // Vectorized Square inside the loop!
            for(size_t idx2 = 0; idx2 <= idx1; ++idx2)
            {
                int c2 = active[idx2];
                double rsq2 = R(i, c2) * R(i, c2);

                SumRsq(c1, c2) += rsq1 * rsq2;
                N(c1, c2) += 1.0;
            }
        }
    }

    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(p, p);
    for(int i=0; i < p; ++i)
    {
        for(int j=0; j <= i; ++j)
        {
            if(N(i,j) > 0.0) 
            {
                double val = SumRsq(i, j) / N(i,j);
                cov(i, j) = val;
                if(i != j) cov(j, i) = val;
            }
        }
    }
    return cov;
}

// Helper to calculate the upper/lower bounds safely
void get_bounds(const Eigen::MatrixXd& cov, const Eigen::ArrayXXd& theta, double& lower, double& upper) 
{
    Eigen::ArrayXXd safe_theta = (theta == 0.0).select(1e-8, theta);
    Eigen::MatrixXd delta = (cov.array() / safe_theta).cwiseAbs().matrix();
    delta.diagonal() = Eigen::VectorXd::Zero(delta.rows());
    
    upper = delta.maxCoeff();
    lower = (delta.array() <= 0.0).select(std::numeric_limits<double>::max(), delta).minCoeff();
    if (lower == std::numeric_limits<double>::max()) lower = 0.0;
}

// [[Rcpp::export]]
bool check_openmp() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}