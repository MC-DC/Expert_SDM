// Option 2 — Weighted likelihood model
// Each expert point contributes to the likelihood proportionally to its
// confidence weight, so low-confidence points have less influence on the
// posterior than high-confidence ones.
//   High -> w = 1.00, Medium -> w = 0.50, Low -> w = 0.25
//
// Note: log_lik in generated quantities uses the unweighted likelihood
// so that LOO-CV evaluates true predictive accuracy, not the weighted score.

data {
  int<lower=0> N;
  array[N] int<lower=0, upper=1> y;   // presence (1) / absence (0)

  vector[N] x1;
  vector[N] x2;

  // Expert values for covariate 1 (scaled units)
  real a1_star;
  real b1_star;
  real c1_star;
  real<lower=0> sigma_a1;
  real<lower=0> sigma_b1;
  real<lower=0> sigma_c1;

  // Expert values for covariate 2
  real a2_star;
  real b2_star;
  real c2_star;
  real<lower=0> sigma_a2;
  real<lower=0> sigma_b2;
  real<lower=0> sigma_c2;

  // Priors on slopes
  real beta1_mu;
  real<lower=0> beta1_sigma;
  real beta2_mu;
  real<lower=0> beta2_sigma;

  // Per-point confidence weights
  vector<lower=0, upper=1>[N] w;
}

parameters {
  real alpha;
  real beta1;
  real beta2;

  // Covariate 1 shape — reparametrised to enforce a1 < c1 < b1
  real a1_raw;
  real log_gap_c1;
  real log_gap_b1;

  // Covariate 2 shape
  real a2_raw;
  real log_gap_c2;
  real log_gap_b2;
}

transformed parameters {
  real a1 = a1_raw;
  real c1 = a1 + exp(log_gap_c1);
  real b1 = c1 + exp(log_gap_b1);

  real a2 = a2_raw;
  real c2 = a2 + exp(log_gap_c2);
  real b2 = c2 + exp(log_gap_b2);

  vector[N] f1;
  vector[N] f2;

  for (n in 1:N) {
    if (x1[n] < a1 || x1[n] > b1)
      f1[n] = 0.0;
    else if (x1[n] <= c1)
      f1[n] = 2.0 * (x1[n] - a1) / ((b1 - a1) * (c1 - a1));
    else
      f1[n] = 2.0 * (b1 - x1[n]) / ((b1 - a1) * (b1 - c1));

    if (x2[n] < a2 || x2[n] > b2)
      f2[n] = 0.0;
    else if (x2[n] <= c2)
      f2[n] = 2.0 * (x2[n] - a2) / ((b2 - a2) * (c2 - a2));
    else
      f2[n] = 2.0 * (b2 - x2[n]) / ((b2 - a2) * (b2 - c2));
  }
}

model {
  // Shape parameter priors
  a1_raw     ~ normal(a1_star, sigma_a1);
  log_gap_c1 ~ normal(log(c1_star - a1_star), sigma_c1);
  log_gap_b1 ~ normal(log(b1_star - c1_star), sigma_b1);

  a2_raw     ~ normal(a2_star, sigma_a2);
  log_gap_c2 ~ normal(log(c2_star - a2_star), sigma_c2);
  log_gap_b2 ~ normal(log(b2_star - c2_star), sigma_b2);

  // Slope priors
  alpha  ~ normal(0, 1);
  beta1  ~ normal(beta1_mu, beta1_sigma);
  beta2  ~ normal(beta2_mu, beta2_sigma);

  // Weighted Bernoulli likelihood:
  // Each observation contributes w[n] times its log-likelihood,
  // so high-confidence points pull the posterior more strongly.
  for (n in 1:N) {
    target += w[n] * bernoulli_logit_lpmf(y[n] | alpha + beta1 * f1[n] + beta2 * f2[n]);
  }
}

generated quantities {
  array[N] int y_rep;
  vector[N] log_lik;
  for (n in 1:N) {
    real mu = alpha + beta1 * f1[n] + beta2 * f2[n];
    y_rep[n]   = bernoulli_logit_rng(mu);
    log_lik[n] = bernoulli_logit_lpmf(y[n] | mu);   // unweighted for LOO
  }
}
