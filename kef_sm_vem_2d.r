# 필요한 라이브러리 로드
library(mvtnorm)
library(ggplot2)
library(gridExtra)

############################################################################
# Part 1: 데이터 생성 함수 모음
# - 원하는 데이터 형태를 함수로 만들어 사용합니다.
# - 새로운 데이터를 테스트하려면 이 섹션에 새 함수를 추가하면 됩니다.
############################################################################

# 1-1. 가우시안 혼합 모델 데이터 (기존 데이터)
generate_gmm_data <- function(n = 200) {
  pi_vals <- c(0.55, 0.45)
  mu <- list(c(-1, -0.5), c(2, 1.2))
  S1 <- matrix(c(0.6, 0.3, 0.3, 0.8), 2, 2)
  S2 <- matrix(c(0.5, -0.2, -0.2, 0.5), 2, 2)
  
  z <- rbinom(n, size = 1, prob = pi_vals[2]) + 1
  X_data <- matrix(NA, n, 2)
  X_data[z == 1, ] <- rmvnorm(sum(z == 1), mean = mu[[1]], sigma = S1)
  X_data[z == 2, ] <- rmvnorm(sum(z == 2), mean = mu[[2]], sigma = S2)
  
  return(data.frame(x1 = X_data[, 1], x2 = X_data[, 2]))
}

# 1-2. 바나나 모양 데이터
generate_banana_data <- function(n = 300) {
  x1 <- rnorm(n, mean = 0, sd = 2)
  x2 <- 0.1 * x1^2 + rnorm(n, mean = 0, sd = 0.8)
  return(data.frame(x1 = x1, x2 = x2))
}

# 1-3. 링(도넛) 모양 데이터
generate_ring_data <- function(n = 400) {
  radius <- rnorm(n, mean = 5, sd = 0.5)
  angle <- runif(n, 0, 2 * pi)
  x1 <- radius * cos(angle)
  x2 <- radius * sin(angle)
  return(data.frame(x1 = x1, x2 = x2))
}


############################################################################
# Part 2: VEM 알고리즘 엔진
# - 이 부분은 데이터 종류와 무관한 핵심 계산 로직입니다.
# - 특별한 경우가 아니면 수정할 필요가 없습니다.
############################################################################

# 커널 및 도함수 정의
rbf_kernel <- function(x, y, sigma) exp(-sum((x - y)^2) / (2 * sigma^2))
grad_x_rbf <- function(x, y, sigma) -rbf_kernel(x, y, sigma) * (x - y) / sigma^2
lap_x_rbf <- function(x, y, sigma) {
  sq_dist <- sum((x - y)^2); d <- length(x)
  rbf_kernel(x, y, sigma) * (sq_dist / sigma^4 - d / sigma^2)
}
grad_sigma_grad_x_rbf <- function(x, y, sigma) {
  rbf <- rbf_kernel(x, y, sigma); sq_dist <- sum((x-y)^2)
  (-rbf * sq_dist / sigma^5 + 2 * rbf / sigma^3) * (x - y)
}
grad_sigma_lap_x_rbf <- function(x, y, sigma) {
  rbf <- rbf_kernel(x, y, sigma); sq_dist <- sum((x - y)^2); d <- length(x)
  grad_sigma_rbf <- rbf * sq_dist / sigma^3
  term1 <- grad_sigma_rbf * (sq_dist / sigma^4 - d/sigma^2)
  term2 <- rbf * (-4*sq_dist/sigma^5 + 2*d/sigma^3)
  return(term1 + term2)
}

# K, b 및 sigma에 대한 도함수들을 계산하는 함수
compute_derivatives <- function(X, basis, sigma) {
  n <- nrow(X); M <- nrow(basis); D <- ncol(X)
  K_sigma <- matrix(0, M, M); b_sigma <- numeric(M)
  dK_dsigma <- matrix(0, M, M); db_dsigma <- numeric(M)
  
  for (i in 1:n) {
    J_i <- matrix(0, D, M); dJ_dsigma_i <- matrix(0, D, M)
    H_i_row <- numeric(M); dH_dsigma_i_row <- numeric(M)
    for (j in 1:M) {
      J_i[, j] <- grad_x_rbf(X[i, ], basis[j, ], sigma)
      dJ_dsigma_i[, j] <- grad_sigma_grad_x_rbf(X[i, ], basis[j, ], sigma)
      H_i_row[j] <- lap_x_rbf(X[i, ], basis[j, ], sigma)
      dH_dsigma_i_row[j] <- grad_sigma_lap_x_rbf(X[i, ], basis[j, ], sigma)
    }
    K_sigma <- K_sigma + t(J_i) %*% J_i
    b_sigma <- b_sigma - H_i_row
    dK_dsigma <- dK_dsigma + (t(dJ_dsigma_i) %*% J_i + t(J_i) %*% dJ_dsigma_i)
    db_dsigma <- db_dsigma - dH_dsigma_i_row
  }
  return(list(K = K_sigma/n, b = b_sigma/n, dK_dsigma = dK_dsigma/n, db_dsigma = db_dsigma/n))
}

# VEM 알고리즘 주 함수
run_vem <- function(X_data, basis_pts, n_iters = 100, lr_sigma = 1e-5) {
  M <- nrow(basis_pts)
  sigma <- 1.0; tau2 <- 1.0; beta <- 1.0
  history <- data.frame(iter = 1:n_iters, elbo = NA, sigma = NA, tau = NA)
  
  cat("VEM 알고리즘 시작...\n")
  for (iter in 1:n_iters) {
     derivs <- compute_derivatives(X_data, basis_pts, sigma)
    K_sigma <- derivs$K; b_sigma <- derivs$b
    
    S <- solve(beta * K_sigma + (1/tau2) * diag(M) + diag(1e-8, M))
    m <- beta * S %*% b_sigma
    
    tau2 <- (sum(diag(S)) + sum(m^2)) / M
    
    grad_L_sigma <- -0.5 * beta * (sum(diag(derivs$dK_dsigma %*% S)) + t(m) %*% derivs$dK_dsigma %*% m) +
      beta * t(derivs$db_dsigma) %*% m
    sigma <- sigma + lr_sigma * grad_L_sigma
    sigma <- max(sigma, 0.01)
    
    log_det_S <- determinant(S, logarithm = TRUE)$modulus[1]
    log_likelihood_term <- -0.5*beta*(sum(diag(K_sigma %*% S)) + t(m)%*%K_sigma%*%m) + beta * t(b_sigma)%*%m
    log_prior_term <- -0.5 * (M * log(tau2) + (sum(diag(S)) + sum(m^2))/tau2)
    entropy_term <- 0.5 * log_det_S
    elbo <- log_likelihood_term + log_prior_term + entropy_term
    
    history[iter, ] <- c(iter, elbo, sigma, sqrt(tau2))
    if (iter %% 100 == 0) cat(sprintf("  Iter %d: ELBO=%.2f, sigma=%.3f, tau=%.3f\n", iter, elbo, sigma, sqrt(tau2)))
  }
  cat("VEM 알고리즘 종료.\n")
  return(list(m = m, S = S, final_params = c(sigma, sqrt(tau2), beta), history = history))
}


############################################################################
# Part 3: 결과 시각화 함수
# - 알고리즘 실행 결과(vem_results)와 원본 데이터(dat)를 받아
#   수렴 과정과 최종 밀도 플롯을 그려줍니다.
############################################################################

visualize_results <- function(dat, vem_results, basis_pts) {
  history <- na.omit(vem_results$history)
  X_data <- as.matrix(dat)
  
  # 1. ELBO 및 하이퍼파라미터 수렴 과정 시각화
  p1 <- ggplot(history, aes(x = iter, y = sigma)) + geom_line(color="#0072B2", size=1) + labs(title = "Sigma (커널 대역폭) 수렴", y = "sigma") + theme_bw()
  p2 <- ggplot(history, aes(x = iter, y = tau)) + geom_line(color="#D55E00", size=1) + labs(title = "Tau (정규화) 수렴", y = "tau") + theme_bw()
  p3 <- ggplot(history, aes(x = iter, y = elbo)) + geom_line(color="#009E73", size=1) + labs(title = "ELBO (목표 함수) 수렴", y = "ELBO") + theme_bw()
  
  print(grid.arrange(p1, p2, p3, ncol = 3))
  
  # 2. 최종 추정된 밀도 플롯
  grid_range <- apply(X_data, 2, function(x) range(x) + c(-1, 1))
  x1_grid <- seq(grid_range[1,1], grid_range[2,1], length.out = 80)
  x2_grid <- seq(grid_range[1,2], grid_range[2,2], length.out = 80)
  eval_grid <- expand.grid(x1 = x1_grid, x2 = x2_grid)

  final_m <- vem_results$m
  final_sigma <- vem_results$final_params[1]
  
  log_density_vals <- apply(eval_grid, 1, function(x_eval) {
    phi_x <- sapply(1:nrow(basis_pts), function(j) rbf_kernel(x_eval, basis_pts[j,], final_sigma))
    sum(final_m * phi_x)
  })
  eval_grid$log_density <- log_density_vals

  p_density <- ggplot() +
    stat_contour(data = eval_grid, aes(x = x1, y = x2, z = log_density, fill = ..level..), geom = "polygon", alpha=0.8) +
    geom_point(data = dat, aes(x = x1, y = x2), alpha = 0.7, size = 1.5, shape=3, color="white") +
    scale_fill_viridis_c(option = "plasma") +
    coord_equal(xlim = grid_range[,1], ylim = grid_range[,2]) +
    labs(title = "VEM으로 최종 학습된 밀도 분포",
         subtitle = paste("최종 sigma =", round(final_sigma, 3), " / 최종 tau =", round(vem_results$final_params[2], 3)),
         x = "x1", y = "x2", fill = "로그 밀도") +
    theme_minimal(base_size = 14)
    
  print(p_density)
}


############################################################################
# Part 4: 메인 스크립트 (사용 예시)
# - 이 부분에서 원하는 데이터 생성 함수를 선택하고,
#   알고리즘을 실행한 뒤, 결과를 시각화합니다.
############################################################################

# --- 1. 데이터 선택 ---
# 아래 주석을 바꾸어 원하는 데이터를 생성하고 테스트해 보세요.
# dat <- generate_gmm_data(n = 200)
dat <- generate_banana_data(n = 300)
# dat <- generate_ring_data(n = 400)

# --- 2. VEM 알고리즘 실행 ---
# 데이터를 matrix 형태로 변환
X_mat <- as.matrix(dat)
# basis point는 데이터 포인트를 그대로 사용 (일반적인 방식)
basis_pts <- X_mat 
# 알고리즘 실행 (반복 횟수와 학습률은 데이터에 따라 조절)
vem_results <- run_vem(X_mat, basis_pts, n_iters = 1000, lr_sigma = 1e-4)

# --- 3. 결과 시각화 ---
visualize_results(dat, vem_results, basis_pts)