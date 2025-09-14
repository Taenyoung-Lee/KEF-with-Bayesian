# kef_scorematching_em.R
# Kernel Exponential Families + Score Matching + Variational EM (R, minimal)
# - RBF kernel (sigma)
# - H_sigma, b_sigma, their sigma-derivatives
# - E-step: q*(alpha) = N(-1/T A^{-1} b, A^{-1})
# - M-step: lambda closed-form; sigma via ELBO gradient ascent on log-sigma
# - Lightweight 2D demo with contour

## ------------------------------
## Utilities
## ------------------------------

pairwise_sq_dists <- function(X, Y) {
  # X: T x d, Y: S x d
  X2 <- rowSums(X^2)
  Y2 <- rowSums(Y^2)
  outer(X2, Y2, "+") - 2 * (X %*% t(Y))
}

rbf_kernel <- function(X, sigma) {
  D2 <- pairwise_sq_dists(X, X)
  K  <- exp(-0.5 * D2 / (sigma^2))
  list(K = K, D2 = D2)
}

grad_k_at_xt <- function(X, t, sigma, K_row_t, D2_row_t) {
  # G_t = [∇_x k(x_t, x_i)]_{i} in shape d x T
  x_t  <- X[t, , drop = FALSE] # 1 x d
  diffs <- matrix(rep(x_t, nrow(X)), nrow = nrow(X), byrow = TRUE) - X  # T x d
  # -(x - x_i)/sigma^2 * k
  Gt <- t(-(t(diffs) / (sigma^2)) * K_row_t) # T x d (before transpose)
  t(Gt) # d x T
}

laplacian_k_at_xt <- function(X, t, sigma, K_row_t, D2_row_t) {
  d <- ncol(X)
  ((-d / (sigma^2)) + (D2_row_t / (sigma^4))) * K_row_t  # length T
}

d_dsigma_K <- function(X, sigma, K, D2) {
  K * (D2 / (sigma^3))
}

d_dsigma_grad_k_at_xt <- function(X, t, sigma, K_row_t, D2_row_t) {
  # ∂σ (∇_x k) = k * (x - x') * (2/σ^3 - ||x-x'||^2 / σ^5)
  x_t  <- X[t, , drop = FALSE]
  diffs <- matrix(rep(x_t, nrow(X)), nrow = nrow(X), byrow = TRUE) - X  # T x d
  factor <- (2.0 / (sigma^3)) - (D2_row_t / (sigma^5))                # length T
  dGt <- (K_row_t * factor) * diffs  # T x d
  t(dGt) # d x T
}

d_dsigma_laplacian_k_at_xt <- function(X, t, sigma, K_row_t, D2_row_t) {
  d <- ncol(X)
  term <- (2*d / (sigma^3)) - ((d + 4) * D2_row_t / (sigma^5)) + ((D2_row_t^2) / (sigma^7))
  K_row_t * term
}

build_terms <- function(X, sigma) {
  Tn <- nrow(X); d <- ncol(X)
  kd <- rbf_kernel(X, sigma)
  K  <- kd$K; D2 <- kd$D2

  H  <- matrix(0, Tn, Tn)
  b  <- rep(0, Tn)
  dK <- d_dsigma_K(X, sigma, K, D2)
  dH <- matrix(0, Tn, Tn)
  db <- rep(0, Tn)

  for (t in 1:Tn) {
    K_row_t  <- K[t, ]
    D2_row_t <- D2[t, ]
    Gt  <- grad_k_at_xt(X, t, sigma, K_row_t, D2_row_t)   # d x T
    dGt <- d_dsigma_grad_k_at_xt(X, t, sigma, K_row_t, D2_row_t) # d x T
    H   <- H + (t(Gt) %*% Gt) / Tn
    dH  <- dH + (t(dGt) %*% Gt + t(Gt) %*% dGt) / Tn
    lap_vec  <- laplacian_k_at_xt(X, t, sigma, K_row_t, D2_row_t)    # T
    dlap_vec <- d_dsigma_laplacian_k_at_xt(X, t, sigma, K_row_t, D2_row_t)
    b  <- b  + lap_vec
    db <- db + dlap_vec
  }
  list(K = K, H = H, b = b, dK = dK, dH = dH, db = db)
}

stable_chol_solve <- function(A, B = diag(nrow(A))) {
  # Solve A X = B with Cholesky + jitter
  jitter <- 1e-8
  for (i in 1:5) {
    AA <- A + jitter * diag(nrow(A))
    ok <- TRUE
    L <- tryCatch(chol(AA), error = function(e) { ok <<- FALSE; NULL })
    if (ok) {
      X <- backsolve(L, forwardsolve(t(L), B))
      return(list(X = X, L = L, jitter = jitter))
    }
    jitter <- jitter * 10
  }
  # fallback
  X <- solve(A, B)
  list(X = X, L = NULL, jitter = jitter)
}

logdet_via_chol <- function(A) {
  cs <- tryCatch(chol(A), error = function(e) NULL)
  if (is.null(cs)) {
    eig <- eigen((A + t(A))/2, symmetric = TRUE, only.values = TRUE)$values
    eig[eig < 1e-12] <- 1e-12
    return(sum(log(eig)))
  } else {
    return(2 * sum(log(diag(cs))))
  }
}

elbo_and_grad_sigma <- function(X, sigma, lam, K, H, b, dK, dH, db) {
  Tn <- nrow(X)
  A  <- H + lam * K
  Ainv <- stable_chol_solve(A)$X

  logdetK <- logdet_via_chol(K + 1e-8 * diag(nrow(K)))
  logdetA <- logdet_via_chol(A + 1e-12 * diag(nrow(A)))

  L <- (Tn/2) * log(lam) + 0.5 * logdetK - 0.5 * logdetA + 0.5/(Tn^2) * as.numeric(t(b) %*% (Ainv %*% b))

  dA <- dH + lam * dK

  # tr(K^{-1} dK)
  Kinv <- stable_chol_solve(K + 1e-8 * diag(nrow(K)))$X
  tr_Kinv_dK <- sum(Kinv * dK) # elementwise sum = trace(Kinv %*% dK)

  tr_Ainv_dA <- sum(Ainv * dA)

  term1 <- 0.5 * tr_Kinv_dK
  term2 <- -0.5 * tr_Ainv_dA
  Ainv_b <- Ainv %*% b
  term3 <- (1.0/(2*Tn^2)) * ( 2.0 * sum(db * as.vector(Ainv_b)) - as.numeric(t(b) %*% (Ainv %*% (dA %*% Ainv_b))) )

  dL <- term1 + term2 + term3
  list(L = L, dL_dsigma = dL, A = A, Ainv = Ainv)
}

update_lambda_closed_form <- function(Tn, K, Ainv, b) {
  tr_term <- sum((K %*% Ainv) * diag(Tn)) # trace(K Ainv)
  # faster: tr(K %*% Ainv) = sum elementwise of (K * t(Ainv))
  tr_term <- sum(K * t(Ainv))
  denom <- tr_term + (1.0 / (Tn^2)) * as.numeric(t(b) %*% (Ainv %*% (K %*% (Ainv %*% b))))
  lam_new <- Tn / max(denom, 1e-12)
  max(1e-8, lam_new)
}

kef_scorematching_em <- function(X, sigma0 = 1.0, lam0 = 1.0, n_iter = 5, step_size = 0.2, verbose = TRUE) {
  Tn <- nrow(X); d <- ncol(X)
  sigma <- sigma0
  lam <- lam0
  history <- list()

  for (it in 1:n_iter) {
    terms <- build_terms(X, sigma)
    K <- terms$K; H <- terms$H; b <- terms$b; dK <- terms$dK; dH <- terms$dH; db <- terms$db

    eg <- elbo_and_grad_sigma(X, sigma, lam, K, H, b, dK, dH, db)
    L <- eg$L; dL_dsigma <- eg$dL_dsigma; Ainv <- eg$Ainv
    alpha_mean <- - (Ainv %*% b) / Tn

    lam_new <- update_lambda_closed_form(Tn, K, Ainv, b)

    log_sigma <- log(sigma)
    log_sigma_new <- log_sigma + step_size * (sigma * dL_dsigma)
    sigma_new <- as.numeric(exp(log_sigma_new))

    history[[it]] <- list(iter = it, sigma = sigma, lambda = lam, ELBO = L,
                          dL_dsigma = dL_dsigma, alpha_norm = sqrt(sum(alpha_mean^2)))

    if (verbose) {
      cat(sprintf("[Iter %02d] ELBO=%.6f  sigma=%.4f->%.4f  lambda=%.3e->%.3e  ||alpha||=%.4f\n",
                  it, L, sigma, sigma_new, lam, lam_new, sqrt(sum(alpha_mean^2))))
    }

    sigma <- sigma_new
    lam   <- lam_new
  }

  # Final alpha*
  terms <- build_terms(X, sigma)
  K <- terms$K; H <- terms$H; b <- terms$b
  A <- H + lam * K
  Ainv <- stable_chol_solve(A)$X
  alpha_star <- - (Ainv %*% b) / Tn

  list(sigma = sigma, lambda = lam, alpha = as.vector(alpha_star), K = K, X = X, history = history)
}

## ------------------------------
## Demo (2D data + contour)
## ------------------------------

set.seed(0)
Tn <- 90; d <- 2
means <- rbind(c(0,0), c(2.5,2.5), c(-2.5,2.5))
cov <- matrix(c(0.25,0,0,0.25), 2, 2)
X <- do.call(rbind, lapply(1:3, function(i) MASS::mvrnorm(n = Tn/3, mu = means[i,], Sigma = cov)))

# Run EM
res <- kef_scorematching_em(X, sigma0 = 0.9, lam0 = 1.0, n_iter = 50, step_size = 0.2, verbose = TRUE)

# Evaluate f*(x) = sum_i alpha_i k(x, x_i) on a grid (only for d=2)
alpha <- res$alpha; sigma <- res$sigma
gx <- seq(min(X[,1]) - 2, max(X[,1]) + 2, length.out = 80)
gy <- seq(min(X[,2]) - 2, max(X[,2]) + 2, length.out = 80)
grid <- as.matrix(expand.grid(gx, gy))
D2g <- pairwise_sq_dists(grid, X)
Kg  <- exp(-0.5 * D2g / (sigma^2))
fgrid <- matrix(as.numeric(Kg %*% alpha), nrow = length(gx), ncol = length(gy))

# Plot
par(mar = c(4,4,2,1))
plot(X, pch = 16, cex = 0.6, col = rgb(0,0,0,0.5), xlab = "x1", ylab = "x2",
     main = "KEF Score-Matching + Variational EM (R demo)")
contour(gx, gy, fgrid, add = TRUE, drawlabels = TRUE)
legend("topright", legend = c("data", "f*(x) contours"), pch = c(16, NA), lty = c(NA,1), bty = "n")
