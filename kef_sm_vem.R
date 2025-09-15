# --- 0. 필요시 패키지 설치 ---
# install.packages("MASS") # ginv 사용을 위해 (필요시)

# --- 1. 데이터 생성 ---
generate_data <- function(n_samples = 150) {
  set.seed(0)
  X1 <- rnorm(n = floor(n_samples * 0.5), mean = -4, sd = 0.5)
  X2 <- rnorm(n = ceiling(n_samples * 0.5), mean = 4, sd = 1.0)
  return(matrix(c(X1, X2), ncol = 1))
}

# --- 2. RBF 커널 및 도함수 구현 ---
# 문서의 (11), (12), (129), (130) 식에 해당

rbf_kernel <- function(X1, X2, sigma) {
  sq_dist_mat <- as.matrix(dist(rbind(X1, X2), method = "euclidean"))^2
  return(exp(-sq_dist_mat[1:nrow(X1), (nrow(X1) + 1):(nrow(X1) + nrow(X2))] / (2 * sigma^2)))
}

grad_k <- function(x, y, sigma) {
  k <- exp(-sum((x - y)^2) / (2 * sigma^2))
  return(- (x - y) / sigma^2 * k)
}

lapl_k <- function(x, y, sigma) {
  d <- length(x)
  diff_norm_sq <- sum((x - y)^2)
  k <- exp(-diff_norm_sq / (2 * sigma^2))
  return((-d / sigma^2 + diff_norm_sq / sigma^4) * k)
}

dsigma_k <- function(x, y, sigma) {
  diff_norm_sq <- sum((x - y)^2)
  k <- exp(-diff_norm_sq / (2 * sigma^2))
  return(k * diff_norm_sq / sigma^3)
}

dsigma_grad_k <- function(x, y, sigma) {
  diff_norm_sq <- sum((x - y)^2)
  k <- exp(-diff_norm_sq / (2 * sigma^2))
  term <- (2 / sigma^3 - diff_norm_sq / sigma^5)
  return(k * (x - y) * term)
}

dsigma_lapl_k <- function(x, y, sigma) {
  d <- length(x)
  diff_norm_sq <- sum((x-y)^2)
  k <- exp(-diff_norm_sq / (2 * sigma^2))
  term1 <- 2 * d / sigma^3
  term2 <- -(d + 4) * diff_norm_sq / sigma^5
  term3 <- diff_norm_sq^2 / sigma^7
  return(k * (term1 + term2 + term3))
}


# --- 3. H, b 행렬 구성 ---
# 문서의 (16), (17) 식에 해당

compute_H_b <- function(X, sigma) {
  T <- nrow(X)
  d <- ncol(X)
  H <- matrix(0, T, T)
  b <- numeric(T)
  
  grad_k_vals <- array(0, dim = c(T, T, d))
  for (t in 1:T) {
    for (i in 1:T) {
      grad_k_vals[t, i, ] <- grad_k(X[t, ], X[i, ], sigma)
    }
  }

  for (i in 1:T) {
    for (j in 1:T) {
      H[i, j] <- sum(sapply(1:T, function(t) grad_k_vals[t, i, ] %*% grad_k_vals[t, j, ]))
    }
  }
  H <- H / T
  
  for (i in 1:T) {
    b[i] <- sum(sapply(1:T, function(t) lapl_k(X[t, ], X[i, ], sigma)))
  }
  
  return(list(H = H, b = b))
}

compute_dsigma_H_b <- function(X, sigma) {
    T <- nrow(X); d <- ncol(X)
    dH <- matrix(0, T, T); db <- numeric(T)

    grad_k_vals <- array(0, dim=c(T, T, d))
    dsigma_grad_k_vals <- array(0, dim=c(T, T, d))
    for (t in 1:T) {
        for (i in 1:T) {
            grad_k_vals[t, i, ] <- grad_k(X[t, ], X[i, ], sigma)
            dsigma_grad_k_vals[t, i, ] <- dsigma_grad_k(X[t, ], X[i, ], sigma)
        }
    }

    for (i in 1:T) {
        for (j in 1:T) {
            term1 <- sum(sapply(1:T, function(t) dsigma_grad_k_vals[t, i, ] %*% grad_k_vals[t, j, ]))
            term2 <- sum(sapply(1:T, function(t) grad_k_vals[t, i, ] %*% dsigma_grad_k_vals[t, j, ]))
            dH[i, j] <- term1 + term2
        }
    }
    dH <- dH / T

    for (i in 1:T) {
        db[i] <- sum(sapply(1:T, function(t) dsigma_lapl_k(X[t, ], X[i, ], sigma)))
    }
    return(list(dH = dH, db = db))
}


# --- 4. EM 알고리즘으로 하이퍼파라미터 튜닝 ---
# 문서의 5.2절, (116), (125), (127) 식에 해당
tune_hyperparams_em <- function(X, n_iter = 20, lr_sigma = 0.01, lambda_cap = 1e5) {
  T <- nrow(X)
  
  # 초기화 (Median Heuristic)
  sigma <- sqrt(0.5 * median(dist(X)^2))
  lmbda <- 0.1
  
  cat(sprintf("초기값: sigma=%.3f, lambda=%.3f\n", sigma, lmbda))

  for (it in 1:n_iter) {
    # --- E-step ---
    K <- rbf_kernel(X, X, sigma)
    H_b <- compute_H_b(X, sigma)
    H <- H_b$H; b <- H_b$b
    
    A <- H + lmbda * K
    A <- A + diag(T) * 1e-6 
    
    # solve(A)에서 에러가 날 경우를 대비한 예외 처리
    A_inv <- tryCatch({
        solve(A)
    }, error = function(e) {
        cat(sprintf("\nIter %d에서 A의 역행렬 계산 실패. 불안정한 업데이트로 인해 중단합니다.\n", it))
        cat("Error: ", e$message, "\n")
        return(NULL)
    })
    
    if (is.null(A_inv)) {
      return(list(sigma = sigma, lmbda = lmbda, error = TRUE))
    }
    
    A_inv_b <- A_inv %*% b

    # --- M-step ---
    # 1. lambda 업데이트
    A_inv_K <- A_inv %*% K
    trace_term <- sum(diag(A_inv_K))
    b_term <- (1/T^2) * t(b) %*% A_inv_K %*% A_inv_b
    lmbda_new <- T / (trace_term + b_term[1, 1])
    
    # --- SOLUTION: 람다에 상한선 설정 ---
    lmbda <- min(lmbda_new, lambda_cap)
    # ------------------------------------

    # 2. sigma 업데이트
    dK <- outer(1:T, 1:T, FUN = Vectorize(function(i, j) dsigma_k(X[i,], X[j,], sigma)))
    d_H_b <- compute_dsigma_H_b(X, sigma)
    dH <- d_H_b$dH; db <- d_H_b$db
    dA <- dH + lmbda * dK
    
    K_reg <- K + diag(T) * 1e-5 
    grad_term1 <- 0.5 * sum(diag(solve(K_reg) %*% dK))
    grad_term2 <- -0.5 * sum(diag(A_inv %*% dA))
    grad_term3 <- (1 / T^2) * (t(db) %*% A_inv_b - 0.5 * t(b) %*% A_inv %*% dA %*% A_inv_b)
    grad_log_sigma <- grad_term1 + grad_term2 + grad_term3[1, 1]
    
    log_sigma <- log(sigma) + lr_sigma * grad_log_sigma
    sigma <- exp(log_sigma)
    
    if (it %% 5 == 0) {
      cat(sprintf("Iter %d/%d: sigma=%.3f, lambda=%.3f\n", it, n_iter, sigma, lmbda))
    }
  }
  return(list(sigma = sigma, lmbda = lmbda, error = FALSE))
}

# --- 5. 최종 모델 학습 및 밀도 추정 ---
score_matching_kef <- function(X, sigma, lmbda) {
  T <- nrow(X)
  K <- rbf_kernel(X, X, sigma)
  H_b <- compute_H_b(X, sigma)
  H <- H_b$H; b <- H_b$b
  
  A <- H + lmbda * K
  A <- A + diag(T) * 1e-6 # Jitter
  
  # alpha 계산 (식 21)
  alpha <- solve(A, -b / T)
  return(alpha)
}

estimate_log_density <- function(x_test, X_train, alpha, sigma) {
  k_vec <- rbf_kernel(x_test, X_train, sigma)
  return(k_vec %*% alpha)
}

# --- 6. 실행 및 시각화 ---
# 1. 데이터 생성
X_train <- generate_data(n_samples = 100)

# 2. EM으로 하이퍼파라미터 튜닝
cat("EM 알고리즘으로 하이퍼파라미터 튜닝 시작...\n")
hyperparams <- tune_hyperparams_em(X_train, n_iter = 20, lr_sigma = 0.005)
sigma_opt <- hyperparams$sigma
lmbda_opt <- hyperparams$lmbda
cat(sprintf("\n최적화된 하이퍼파라미터: sigma = %.3f, lambda = %.3f\n", sigma_opt, lmbda_opt))

# 3. 최적 파라미터로 alpha* 계산
alpha_opt <- score_matching_kef(X_train, sigma_opt, lmbda_opt)

# 4. 테스트 데이터에 대한 밀도 추정
x_test <- matrix(seq(-8, 8, length.out = 400), ncol = 1)
log_p_unnormalized <- estimate_log_density(x_test, X_train, alpha_opt, sigma_opt)
p_unnormalized <- exp(log_p_unnormalized - max(log_p_unnormalized)) # 수치 안정화

# 5. 시각화
hist(X_train, breaks = 20, freq = FALSE, main = "KEF Density Estimation via Score Matching (R)",
     xlab = "x", ylab = "Density", col = "lightblue", border = "white")

lines(x_test, p_unnormalized, col = "red", lwd = 2)

# 실제 분포(참고용)
true_pdf <- function(x) {
  0.5 * dnorm(x, mean = -4, sd = 0.5) + 0.5 * dnorm(x, mean = 4, sd = 1.0)
}
lines(x_test, true_pdf(x_test), col = "black", lty = 2, lwd = 2)

legend("topright", 
       legend = c("Data Histogram", "KEF Estimated Density", "True Density"),
       col = c("lightblue", "red", "black"),
       lty = c(1, 1, 2),
       lwd = c(10, 2, 2))