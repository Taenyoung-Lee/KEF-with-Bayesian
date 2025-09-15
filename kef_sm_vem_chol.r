### ------------------------------------------------------------
### KEF + Score Matching + Variational EM (R, 안정화 패치)
### - Cholesky 기반 선형계 풀이 (역행렬 금지)
### - H/dH 대칭화
### - 최종 밀도 정규화(수치적 적분)
### ------------------------------------------------------------

# --- 0. 필요시 패키지 설치 ---
# install.packages("MASS") # (ginv 필요시)

# --- 1. 데이터 생성 ---
generate_data <- function(n_samples = 150) {
  set.seed(0)
  X1 <- rnorm(n = floor(n_samples * 0.5), mean = -4, sd = 0.5)
  X2 <- rnorm(n = ceiling(n_samples * 0.5), mean = 4, sd = 1.0)
  matrix(c(X1, X2), ncol = 1)
}

# --- 2. RBF 커널 및 도함수 ---
rbf_kernel <- function(X1, X2, sigma) {
  sq_dist_mat <- as.matrix(dist(rbind(X1, X2), method = "euclidean"))^2
  exp(-sq_dist_mat[1:nrow(X1), (nrow(X1) + 1):(nrow(X1) + nrow(X2))] / (2 * sigma^2))
}

grad_k <- function(x, y, sigma) {
  k <- exp(-sum((x - y)^2) / (2 * sigma^2))
  -(x - y) / sigma^2 * k
}

lapl_k <- function(x, y, sigma) {
  d <- length(x)
  diff_norm_sq <- sum((x - y)^2)
  k <- exp(-diff_norm_sq / (2 * sigma^2))
  (-d / sigma^2 + diff_norm_sq / sigma^4) * k
}

# --- sigma 미분들 ---
dsigma_k <- function(x, y, sigma) {
  diff_norm_sq <- sum((x - y)^2)
  k <- exp(-diff_norm_sq / (2 * sigma^2))
  k * diff_norm_sq / sigma^3
}

dsigma_grad_k <- function(x, y, sigma) {
  diff_norm_sq <- sum((x - y)^2)
  k <- exp(-diff_norm_sq / (2 * sigma^2))
  term <- (2 / sigma^3 - diff_norm_sq / sigma^5)
  k * (x - y) * term
}

dsigma_lapl_k <- function(x, y, sigma) {
  d <- length(x)
  diff_norm_sq <- sum((x-y)^2)
  k <- exp(-diff_norm_sq / (2 * sigma^2))
  term1 <- 2 * d / sigma^3
  term2 <- -(d + 4) * diff_norm_sq / sigma^5
  term3 <- diff_norm_sq^2 / sigma^7
  k * (term1 + term2 + term3)
}

# --- 3. H, b 구성 ---
compute_H_b <- function(X, sigma) {
  T <- nrow(X); d <- ncol(X)
  H <- matrix(0, T, T)
  b <- numeric(T)

  grad_k_vals <- array(0, dim = c(T, T, d))
  for (t in 1:T) {
    for (i in 1:T) grad_k_vals[t, i, ] <- grad_k(X[t, ], X[i, ], sigma)
  }

  for (i in 1:T) for (j in 1:T) {
    H[i, j] <- sum(sapply(1:T, function(t) grad_k_vals[t, i, ] %*% grad_k_vals[t, j, ]))
  }
  H <- H / T
  H <- 0.5 * (H + t(H))   # 대칭화

  for (i in 1:T) {
    b[i] <- sum(sapply(1:T, function(t) ds <- lapl_k(X[t, ], X[i, ], sigma)))
  }
  list(H = H, b = b)
}

compute_dsigma_H_b <- function(X, sigma) {
  T <- nrow(X); d <- ncol(X)
  dH <- matrix(0, T, T); db <- numeric(T)

  grad_k_vals <- array(0, dim = c(T, T, d))
  dsigma_grad_k_vals <- array(0, dim = c(T, T, d))
  for (t in 1:T) {
    for (i in 1:T) {
      grad_k_vals[t, i, ] <- grad_k(X[t, ], X[i, ], sigma)
      dsigma_grad_k_vals[t, i, ] <- dsigma_grad_k(X[t, ], X[i, ], sigma)
    }
  }
  for (i in 1:T) for (j in 1:T) {
    term1 <- sum(sapply(1:T, function(t) dsigma_grad_k_vals[t, i, ] %*% grad_k_vals[t, j, ]))
    term2 <- sum(sapply(1:T, function(t) grad_k_vals[t, i, ] %*% dsigma_grad_k_vals[t, j, ]))
    dH[i, j] <- term1 + term2
  }
  dH <- dH / T
  dH <- 0.5 * (dH + t(dH)) # 대칭화

  for (i in 1:T) {
    db[i] <- sum(sapply(1:T, function(t) dsigma_lapl_k(X[t, ], X[i, ], sigma)))
  }
  list(dH = dH, db = db)
}

# --- 3.5 선형계 풀이 유틸 (Cholesky) ---
chol_solve <- function(A, B) {
  # A X = B 를 X로 푸는 함수 (A: SPD 가정)
  L <- chol(A)
  X <- backsolve(L, forwardsolve(t(L), B))
  X
}

# --- 4. EM으로 하이퍼파라미터 튜닝 ---
tune_hyperparams_em <- function(X, n_iter = 20, lr_sigma = 0.01,
                                lambda_cap = 1e5, lambda_floor = 1e-8,
                                jitter_A = 1e-6, jitter_K = 1e-8) {
  T <- nrow(X)

  # 초기화: Median Heuristic
  sigma <- sqrt(0.5 * median(dist(X)^2))
  lmbda <- 0.1

  cat(sprintf("초기값: sigma=%.4f, lambda=%.4f\n", sigma, lmbda))

  for (it in 1:n_iter) {
    # --- E-step 준비 ---
    K <- rbf_kernel(X, X, sigma)
    Hb <- compute_H_b(X, sigma); H <- Hb$H; b <- Hb$b

    A <- H + lmbda * K
    diag(A) <- diag(A) + jitter_A

    # Cholesky 실패 시 중단
    L_A <- tryCatch(chol(A), error = function(e) NULL)
    if (is.null(L_A)) {
      cat(sprintf("\nIter %d: chol(A) 실패. sigma=%.4f, lambda=%.4f\n", it, sigma, lmbda))
      return(list(sigma = sigma, lmbda = lmbda, error = TRUE))
    }

    # A^{-1}b, A^{-1}K (역행렬 없이)
    Ainv_b <- backsolve(L_A, forwardsolve(t(L_A), b))
    Ainv_K <- backsolve(L_A, forwardsolve(t(L_A), K))

    # --- M-step: lambda 업데이트 (폐형식) ---
    trace_term <- sum(diag(Ainv_K))
    b_term <- as.numeric((t(b) %*% Ainv_K %*% Ainv_b) / T^2)
    lmbda_new <- T / (trace_term + b_term)

    # 안정화 클리핑
    lmbda <- max(min(lmbda_new, lambda_cap), lambda_floor)

    # --- M-step: sigma 업데이트 (ELBO 기울기) ---
    # dK, dH, db
    dK <- outer(1:T, 1:T, FUN = Vectorize(function(i, j) dsigma_k(X[i,], X[j,], sigma)))
    dHb <- compute_dsigma_H_b(X, sigma); dH <- dHb$dH; db <- dHb$db
    dA <- dH + lmbda * dK

    # grad_term1 = 0.5 * tr(K^{-1} dK)  (역행렬 대신 선형계 풀이)
    K_reg <- K; diag(K_reg) <- diag(K_reg) + jitter_K
    L_K <- tryCatch(chol(K_reg), error = function(e) NULL)
    if (is.null(L_K)) {
      cat(sprintf("\nIter %d: chol(K) 실패(σ 그라디언트 계산). sigma=%.4f\n", it, sigma))
      return(list(sigma = sigma, lmbda = lmbda, error = TRUE))
    }
    # K^{-1} dK의 trace: K X = dK 풀어서 X = K^{-1} dK, trace = sum(diag(X))
    X_K <- backsolve(L_K, forwardsolve(t(L_K), dK))
    grad_term1 <- 0.5 * sum(diag(X_K))

    # grad_term2 = -0.5 * tr(A^{-1} dA)  (A Y = dA 풀어 Y = A^{-1} dA)
    Y_A <- backsolve(L_A, forwardsolve(t(L_A), dA))
    grad_term2 <- -0.5 * sum(diag(Y_A))

    # grad_term3 = (1/T^2) * ( db^T A^{-1} b - 0.5 b^T A^{-1} dA A^{-1} b )
    Ainv_db <- backsolve(L_A, forwardsolve(t(L_A), db))
    tmp <- backsolve(L_A, forwardsolve(t(L_A), dA %*% Ainv_b))
    grad_term3 <- (1 / T^2) * ( as.numeric(t(db) %*% Ainv_b) - 0.5 * as.numeric(t(b) %*% tmp) )

    grad_log_sigma <- grad_term1 + grad_term2 + grad_term3

    # 로그-스케일 업데이트 (항상 양수 유지)
    log_sigma <- log(sigma) + lr_sigma * grad_log_sigma
    sigma <- exp(log_sigma)

    if (it %% 5 == 0) {
      cat(sprintf("Iter %2d/%d: sigma=%.4f, lambda=%.5f | grad_log_sigma=%.4e\n",
                  it, n_iter, sigma, lmbda, grad_log_sigma))
    }
  }

  list(sigma = sigma, lmbda = lmbda, error = FALSE)
}

# --- 5. 최종 모델 학습 및 밀도 추정 ---
score_matching_kef <- function(X, sigma, lmbda, jitter_A = 1e-6) {
  T <- nrow(X)
  K <- rbf_kernel(X, X, sigma)
  Hb <- compute_H_b(X, sigma); H <- Hb$H; b <- Hb$b
  A <- H + lmbda * K
  diag(A) <- diag(A) + jitter_A

  # alpha = -(A^{-1} b)/T  (Cholesky)
  L <- chol(A)
  Ainv_b <- backsolve(L, forwardsolve(t(L), b))
  alpha <- - Ainv_b / T
  as.numeric(alpha)
}

estimate_log_density <- function(x_test, X_train, alpha, sigma) {
  k_vec <- rbf_kernel(x_test, X_train, sigma)
  as.numeric(k_vec %*% alpha)
}

# --- 6. 실행 및 시각화 ---
# 1) 데이터
X_train <- generate_data(n_samples = 100)

# 2) 하이퍼파라미터 튜닝
cat("EM 알고리즘으로 하이퍼파라미터 튜닝 시작...\n")
hyper <- tune_hyperparams_em(
  X_train, n_iter = 20, lr_sigma = 0.005,
  lambda_cap = 1e5, lambda_floor = 1e-8
)
stopifnot(!hyper$error)
sigma_opt <- hyper$sigma
lmbda_opt <- hyper$lmbda
cat(sprintf("\n최적화된 하이퍼파라미터: sigma=%.4f, lambda=%.6f\n", sigma_opt, lmbda_opt))

# 3) alpha*
alpha_opt <- score_matching_kef(X_train, sigma_opt, lmbda_opt)

# 4) 테스트 점수/밀도
x_test <- matrix(seq(-8, 8, length.out = 400), ncol = 1)
log_p_unn <- estimate_log_density(x_test, X_train, alpha_opt, sigma_opt)

# 수치 안정화 + 정규화 (∫ p = 1)
p_unn <- exp(log_p_unn - max(log_p_unn))
dx <- diff(range(x_test[,1]))/(nrow(x_test)-1)
Zhat <- sum(p_unn) * dx
p_hat <- p_unn / Zhat

# 5) 시각화
hist(as.numeric(X_train), breaks = 20, freq = FALSE,
     main = "KEF Density Estimation via Score Matching (R, 안정화 패치)",
     xlab = "x", ylab = "Density", col = "lightblue", border = "white")
lines(x_test, p_hat, col = "red", lwd = 2)

# 참고용: 실제 혼합분포
true_pdf <- function(x) 0.5 * dnorm(x, mean = -4, sd = 0.5) + 0.5 * dnorm(x, mean = 4, sd = 1.0)
lines(x_test, true_pdf(x_test), col = "black", lty = 2, lwd = 2)

legend("topright",
       legend = c("Data Histogram", "KEF Estimated Density (normalized)", "True Density"),
       col = c("lightblue", "red", "black"),
       lty = c(1, 1, 2), lwd = c(10, 2, 2))
