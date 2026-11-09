// The solvers of the nuclear-norm-penalized GLM estimator (see R/estimator.R): the proximal map, FISTA and proximal
// Newton.  Each function follows the Python package's estimator.py line by line.
//
// The domain C_R arrives as `radius` (empty: no domain; otherwise the radii R_1, ..., R_G of the main-effect groups
// and R_Theta last, NaN for no ball) and `sizes` (the sizes of the main-effect groups; empty when there are none).

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

static double nuclear_norm(const VectorXd& theta, int d1, int d2) {
  Eigen::Map<const MatrixXd> Theta(theta.data() + theta.size() - d1 * d2, d1, d2);
  return Eigen::BDCSVD<MatrixXd>(Theta).singularValues().sum();
}

// The largest eigenvalue of the symmetric matrix G.
// [[Rcpp::export]]
double top_eigenvalue(const Eigen::MatrixXd& G) {
  return Eigen::SelfAdjointEigenSolver<MatrixXd>(G, Eigen::EigenvaluesOnly).eigenvalues().maxCoeff();
}

// Proximal map of step * lam ||Theta||_* on C_R: the singular values of the interaction block are soft-thresholded at
// step * lam and scaled into the Frobenius ball; each main-effect group is projected onto its ball.
static VectorXd prox(const VectorXd& u, double step, double lam, int d1, int d2, const VectorXd& radius,
                     const VectorXi& sizes) {
  int p0 = u.size() - d1 * d2;
  Eigen::Map<const MatrixXd> Theta(u.data() + p0, d1, d2);
  Eigen::BDCSVD<MatrixXd> svd(Theta, Eigen::ComputeThinU | Eigen::ComputeThinV);
  VectorXd sv = (svd.singularValues().array() - step * lam).max(0.0).matrix();
  if (radius.size() > 0) {
    double RT = radius(radius.size() - 1);
    if (!std::isnan(RT) && sv.norm() > RT) sv *= RT / sv.norm();
  }
  VectorXd out(u.size());
  MatrixXd inter = svd.matrixU() * sv.asDiagonal() * svd.matrixV().transpose();
  out.tail(d1 * d2) = Eigen::Map<VectorXd>(inter.data(), d1 * d2);
  int start = 0;
  for (int g = 0; g < sizes.size(); g++) {
    VectorXd block = u.segment(start, sizes(g));
    if (radius.size() > 0 && !std::isnan(radius(g)) && block.norm() > radius(g)) block *= radius(g) / block.norm();
    out.segment(start, sizes(g)) = block;
    start += sizes(g);
  }
  return out;
}

// min 1/2 x'Gx - b'x + lam ||Theta(x)||_* on C_R by FISTA with adaptive restart, started at x0; L the largest
// eigenvalue of G.
// [[Rcpp::export]]
Eigen::VectorXd fista_cpp(const Eigen::MatrixXd& G, const Eigen::VectorXd& b, double lam, int d1, int d2,
                          const Eigen::VectorXd& x0, double L, int iters, double tol, const Eigen::VectorXd& radius,
                          const Eigen::VectorXi& sizes) {
  L = std::max(L, 1e-12);
  VectorXd x = x0, y = x0, xn;
  double tk = 1.0;
  for (int i = 0; i < iters; i++) {
    xn = prox(y - (G * y - b) / L, 1.0 / L, lam, d1, d2, radius, sizes);
    if ((L * (y - xn)).norm() <= tol * std::max(1.0, L * xn.norm())) return xn;
    if ((y - xn).dot(xn - x) > 0) tk = 1.0;                                  // adaptive restart
    double tn = (1.0 + std::sqrt(1.0 + 4.0 * tk * tk)) / 2.0;
    y = xn + ((tk - 1.0) / tn) * (xn - x);
    x = xn;
    tk = tn;
  }
  return x;
}

static VectorXd sigmoid(const VectorXd& u) { return (0.5 * (1.0 + (0.5 * u.array()).tanh())).matrix(); }

// min (1/s) sum_i [trials log(1 + e^{z_i'x}) - y_i z_i'x] + lam ||Theta(x)||_* on C_R by proximal Newton.  H, the
// curvature-weighted Gram matrix, may be passed in (0 x 0: none), with its top eigenvalue LH (NaN: compute it); H is
// rebuilt after a backtracked step, when the residual fails to halve, and after eight steps.
// [[Rcpp::export]]
Eigen::VectorXd proximal_newton_cpp(const Eigen::MatrixXd& Z, const Eigen::VectorXd& y, double lam, int d1, int d2,
                                    const Eigen::VectorXd& x0, int trials, const Eigen::VectorXd& radius,
                                    const Eigen::VectorXi& sizes, Eigen::MatrixXd H, double LH, double L_max, int iters,
                                    double tol, int inner_iters, double inner_tol) {
  const double s = y.size(), nt = trials;
  L_max = std::max(L_max, 1e-12);
  auto objective = [&](const VectorXd& u, double nuc) {                       // u = Z x, nuc = ||Theta(x)||_*
    Eigen::ArrayXd softplus = u.array().max(0.0) + (-u.array().abs()).exp().log1p();
    return (nt * softplus - y.array() * u.array()).mean() + lam * nuc;
  };
  VectorXd x = x0, u = Z * x;
  double nuc = nuclear_norm(x, d1, d2);
  double f = objective(u, nuc);                                               // u, nuc and f go with x
  bool refresh = H.size() == 0;
  double res_old = INFINITY;
  for (int it = 1; it <= iters; it++) {
    VectorXd g = Z.transpose() * (nt * sigmoid(u) - y) / s;
    VectorXd xp = prox(x - g / L_max, 1.0 / L_max, lam, d1, d2, radius, sizes);
    double res = (L_max * (x - xp)).norm() / std::max(1.0, L_max * xp.norm());
    if (res <= tol) return xp;
    refresh = refresh || res > 0.5 * res_old || it > 8;
    res_old = res;
    if (refresh) {
      VectorXd q = sigmoid(u);
      VectorXd w = (nt * q.array() * (1.0 - q.array())).matrix();
      H = Z.transpose() * (Z.array().colwise() * w.array()).matrix() / s;
      LH = top_eigenvalue(H);
    }
    if (std::isnan(LH)) LH = top_eigenvalue(H);
    VectorXd v = fista_cpp(H, H * x - g, lam, d1, d2, x, LH, inner_iters, std::max(inner_tol, 0.1 * res), radius,
                           sizes);
    VectorXd step = v - x;
    double alpha = 1.0;
    VectorXd xn = v, un = Z * v;
    double nucn = nuclear_norm(v, d1, d2);
    double model = g.dot(step) + lam * (nucn - nuc);
    double fn = objective(un, nucn);
    while (fn > f + 1e-4 * alpha * model && alpha > 1e-3) {
      alpha *= 0.5;
      xn = x + alpha * step;
      un = Z * xn;
      nucn = nuclear_norm(xn, d1, d2);
      fn = objective(un, nucn);
    }
    x = xn;
    u = un;
    nuc = nucn;
    f = fn;
    refresh = alpha < 1.0;
  }
  VectorXd g = Z.transpose() * (nt * sigmoid(u) - y) / s;
  return prox(x - g / L_max, 1.0 / L_max, lam, d1, d2, radius, sizes);
}
