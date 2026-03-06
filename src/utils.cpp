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

//vestigial, TODO: still used in algo 1 - gotta fix that
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
                    Eigen::MatrixXd & Et,
                    const Eigen::Ref<const Eigen::MatrixXi> & MAP, 
                    int i, int k, int t, int kt)
{
    // Function body remains identical
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

void Z_assemble_IP(const Eigen::Ref<const Eigen::MatrixXd> & masterZ, 
                   Eigen::MatrixXd & Z_out,
                   const Eigen::Ref<const Eigen::MatrixXi> & MAP,
                   int i, int k, int t, int kt)
{
    // Function body remains identical
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

void U_assemble_IP(const Eigen::Ref<const Eigen::MatrixXd> & masterU, 
                   Eigen::MatrixXd & U_out, 
                   const Eigen::Ref<const Eigen::MatrixXi> & MAP,
                   int i, int p, int k, int t, int kt)
{
    /*
        masterU comes in as t x p
        U_out goes out as kt x p
    */
    if(p<=0){return;}
    U_out.resize(kt,p);
    // 1. Map the absolute time_idx (0 to t-1) to the compressed masterU row (0 to t_i-1)
    std::vector<int> time_to_u_row(t, -1);
    int u_row = 0;
    for(int time_idx = 0; time_idx < t; ++time_idx) {
        // Check if this timepoint has data for AT LEAST one node
        for(int node = 0; node < k; ++node) {
            if(MAP(i, node * t + time_idx) == 1) {
                time_to_u_row[time_idx] = u_row++;
                break; // Found it, move to next time_idx
            }
        }
    }

    // 2. Assemble Ui in the EXACT SAME ORDER as Xi and Zi
    int local_row = 0;
    for(int j = 0; j < k * t; ++j) {
        if(MAP(i, j) == 1) {
            int time_idx = j % t; // Extract the absolute time
            // Grab the correct compressed row for this timepoint
            U_out.row(local_row) = masterU.row(time_to_u_row[time_idx]);
            local_row++;
        }
    }
}

Eigen::VectorXd update_residuals(const Eigen::MatrixXd & X,
                                const Eigen::MatrixXd & U, 
                                const Eigen::VectorXd & y, 
                                const Eigen::VectorXd & beta, 
                                const Eigen::VectorXi & kt_vec,
                                const Eigen::MatrixXi & MAP,
                                int n, int k, int t, int nkt)
{
    int q = X.cols();
    int p = U.cols();
    int cnt = 0;
    int cnt2 = 0;
    Eigen::VectorXd r(nkt);
    Eigen::MatrixXd Ui, Xi;
    if(p>0)
    {
        Ui = Eigen::MatrixXd::Zero(k*t,p);
    }
    for(int i = 0; i < n; ++i) 
    {
        int kt = kt_vec(i);
        if(kt == 0){continue;}
        int t0 = kt % k;
        if(p>0) {U_assemble_IP(U(Eigen::seqN(cnt2,t0),Eigen::all),Ui,MAP, i, p,k,t,kt);}
        Xi = X.block(cnt, 0, kt, q);
        r.segment(cnt,kt) = y.segment(cnt,kt) - Xi * beta.segment(0,q);
        if(p>0){ r.segment(cnt,kt) -= Ui * beta.segment(q,p);}

        cnt += kt;
        cnt2 += t0;
    }
    return(r);
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

void estimate_beta2(const Eigen::Ref<const Eigen::MatrixXd> & X, 
                    const Eigen::Ref<const Eigen::VectorXd> & y, 
                    const Eigen::Ref<const Eigen::MatrixXd> & Z,
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

    Eigen::MatrixXd Zi, Xi, ZiX, M;
    Eigen::VectorXd yi, E_inv, Ziy;
    int cnt = 0;
    
    for(int i = 0; i < n; ++i)
    {
        int kt = kt_vec(i);
        
        // Get Z for this subject
        Z_assemble_IP(Z, Zi, MAP, i, k, t, kt);
        
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
        ZiX = Zi.transpose() * X_tilde;  
        Ziy = Zi.transpose() * y_tilde;  
        
        // Inner matrix M = D^-1 + Z^T * E^-1 * Z  (Always exactly 2k x 2k)
        M = D_inv + Zi.transpose() * E_inv.asDiagonal() * Zi; 
        
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

void estimate_beta3(const Eigen::Ref<const Eigen::MatrixXd> & X, 
                    const Eigen::Ref<const Eigen::MatrixXd> & U, 
                    const Eigen::Ref<const Eigen::VectorXd> & y, 
                    const Eigen::Ref<const Eigen::MatrixXd> & Z,
                    const Eigen::Ref<const Eigen::MatrixXd> & D,
                    const Eigen::Ref<const Eigen::VectorXd> & E,
                    const Eigen::VectorXi & kt_vec, 
                    const Eigen::Ref<const Eigen::MatrixXi> & MAP,
                    Eigen::VectorXd & beta,
                    int n, int k, int t)
{
    int p = U.cols();
    int q = X.cols();
    Eigen::MatrixXd XVX = Eigen::MatrixXd::Zero(q,q);
    Eigen::VectorXd XVy = Eigen::VectorXd::Zero(q);
    Eigen::MatrixXd UVX;
    Eigen::MatrixXd UVU;
    Eigen::VectorXd UVy;
    if(p>0)
    {    
        UVX = Eigen::MatrixXd::Zero(p,q);
        UVU = Eigen::MatrixXd::Zero(p,p);
        UVy = Eigen::VectorXd::Zero(p);
    }
    // W is the full fixeff matrix, W = [Z X]

    // 1. Invert D EXACTLY ONCE outside the loop (Massive CPU saving!)
    Eigen::MatrixXd D_inv;
    Eigen::LDLT<Eigen::MatrixXd> ldlt_D(D);
    if(ldlt_D.info() == Eigen::Success) {
        D_inv = ldlt_D.solve(Eigen::MatrixXd::Identity(2*k, 2*k));
    } else {
        D_inv = D.completeOrthogonalDecomposition().pseudoInverse();
    }

    // --- HOISTED ALLOCATIONS (Zero memory requested inside the loop) ---
    Eigen::MatrixXd Zi, Xi, ZiX, M, ZiU, Ui;
    Eigen::VectorXd yi, E_inv(k*t), Ziy;
    
    // Pre-allocate maximum possible sizes
   // Pre-allocate maximum possible sizes
    Eigen::MatrixXd X_tilde(k*t, q), Zi_tilde(k*t, 2*k);
    Eigen::VectorXd y_tilde(k*t);
    
    // PRE-SIZE THESE SO .noalias() DOESN'T HIT NIL
    Eigen::MatrixXd MZiX(2*k, q); 
    Eigen::VectorXd MZiy(2*k);    
    
    Eigen::MatrixXd U_tilde;
    if (p > 0) {
        U_tilde.resize(k*t, p);
    }
    
    int cnt = 0;
    int cnt2 = 0;
    
    for(int i = 0; i < n; ++i)
    {
        int kt = kt_vec(i);
        if(kt == 0) continue;
        int t0 = kt / k; // FIXED: / instead of %
        
        Z_assemble_IP(Z, Zi, MAP, i, k, t, kt); 
        
        int cnt3 = 0, c = 0;
        for(int j = 0; j < k; ++j) {
            for(int l = 0; l < t; ++l) {
                if(MAP(i, cnt3) == 1) E_inv(c++) = 1.0 / E(j); 
                cnt3++;
            }
        }
        
        Xi = X.block(cnt, 0, kt, q);
        yi = y.segment(cnt, kt);
        
        // 1. PURE SIMD SCALING (No dense diagonal matrices!)
        auto E_inv_head = E_inv.head(kt).array();
        X_tilde.topRows(kt) = Xi.array().colwise() * E_inv_head;
        y_tilde.head(kt) = yi.array() * E_inv_head;
        Zi_tilde.topRows(kt) = Zi.array().colwise() * E_inv_head;
        
        XVX.noalias() += Xi.transpose() * X_tilde.topRows(kt);
        XVy.noalias() += Xi.transpose() * y_tilde.head(kt);

        if (p > 0) {
            U_assemble_IP(U(Eigen::seqN(cnt2,t0),Eigen::all), Ui, MAP, i, p, k,t, kt); 
            U_tilde.topRows(kt) = Ui.array().colwise() * E_inv_head;
            
            UVX.noalias() += Ui.transpose() * X_tilde.topRows(kt);
            UVU.noalias() += Ui.transpose() * U_tilde.topRows(kt);
            UVy.noalias() += Ui.transpose() * y_tilde.head(kt);
            ZiU.noalias() = Zi.transpose() * U_tilde.topRows(kt);
        }
        
        ZiX.noalias() = Zi.transpose() * X_tilde.topRows(kt);  
        Ziy.noalias() = Zi.transpose() * y_tilde.head(kt);
        
        // 2. REUSE THE SCALED Zi FOR M
        M = D_inv;
        M.noalias() += Zi.transpose() * Zi_tilde.topRows(kt); 
        
        Eigen::LDLT<Eigen::MatrixXd> ldlt_M(M);
        if(ldlt_M.info() == Eigen::Success) {
            
            // 1. Copy the Right-Hand Sides into the pre-allocated buffers
            MZiy = Ziy;
            MZiX = ZiX;
            
            // 2. Perform the forward/backward substitution directly on the memory bytes
            ldlt_M.solveInPlace(MZiy);
            ldlt_M.solveInPlace(MZiX);
            
            // 3. Subtract from the accumulators
            XVX.noalias() -= ZiX.transpose() * MZiX;
            XVy.noalias() -= ZiX.transpose() * MZiy;
            
            if (p > 0) {
                UVX.noalias() -= ZiU.transpose() * MZiX;
                UVy.noalias() -= ZiU.transpose() * MZiy;
                
                // For UVU, we can do the same trick!
                Eigen::MatrixXd MZiU = ZiU;
                ldlt_M.solveInPlace(MZiU);
                UVU.noalias() -= ZiU.transpose() * MZiU;
            }
        } else {
            // Fallback for singular matrices
            Eigen::MatrixXd M_inv = M.completeOrthogonalDecomposition().pseudoInverse();
            MZiX.noalias() = M_inv * ZiX; // Now safe because MZiX is sized (2k x q)
            
            XVX.noalias() -= ZiX.transpose() * MZiX;
            XVy.noalias() -= ZiX.transpose() * (M_inv * Ziy);
            
            if (p > 0) {
                // Drop .noalias() so Eigen automatically sizes this to (2k x p)
                Eigen::MatrixXd MZiU = M_inv * ZiU; 
                
                UVX.noalias() -= ZiU.transpose() * MZiX;
                UVy.noalias() -= ZiU.transpose() * (M_inv * Ziy);
                UVU.noalias() -= ZiU.transpose() * MZiU;
            }
        }
        cnt += kt;
        cnt2 += t0;
    }

    // --- FINAL SOLVE BLOCK ---
    if (p == 0) {
        // Fall back to standard GLS if no U covariates
        Eigen::LDLT<Eigen::MatrixXd> ldlt_XVX(XVX);
        if(ldlt_XVX.info() == Eigen::Success)
        {
            beta = ldlt_XVX.solve(XVy);
        }
        else
        {
            beta = XVX.completeOrthogonalDecomposition().pseudoInverse() * XVy;
        }
    } else {
        // Assemble the full augmented system
        Eigen::MatrixXd WVW(p + q, p + q);
        WVW.block(0, 0, q, q) = XVX;
        WVW.block(q, q, p, p) = UVU;
        WVW.block(0, q, q, p) = UVX.transpose();
        WVW.block(q, 0, p, q) = UVX;
        
        Eigen::VectorXd WVy(p + q);
        WVy << XVy, UVy;

        // TODO: block inverse
        beta = WVW.ldlt().solve(WVy);
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

// [[Rcpp::export]]
bool check_openmp() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}