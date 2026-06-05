//

data {
  int<lower=0> N;
  array[N] int<lower=0, upper=1> y;   // presence (1) / absence (0)

  vector[N] x1;                       
  vector[N] x2;                        

  // Expert values for covariate 1 (on scaled units)
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

}

parameters {
  real alpha;
  real beta1;
  real beta2;

  // Covariate 1 shape — reparametrized to enforce a1 < c1 < b1
  real a1_raw;
  real log_gap_c1;   // log(c1 - a1)
  real log_gap_b1;   // log(b1 - c1)

  // Covariate 2 shape
  real a2_raw;
  real log_gap_c2;
  real log_gap_b2;
}

transformed parameters {
  // Recover ordered (a, c, b) for each covariate
  real a1 = a1_raw;
  real c1 = a1 + exp(log_gap_c1);
  real b1 = c1 + exp(log_gap_b1);

  real a2 = a2_raw;
  real c2 = a2 + exp(log_gap_c2);
  real b2 = c2 + exp(log_gap_b2);

  vector[N] f1;
  vector[N] f2;

  for (n in 1:N) {
    // Triangular transform — covariate 1
    if (x1[n] < a1 || x1[n] > b1)
      f1[n] = 0.0;
    else if (x1[n] <= c1)
      f1[n] = 2.0 * (x1[n] - a1) / ((b1 - a1) * (c1 - a1));
    else
      f1[n] = 2.0 * (b1 - x1[n]) / ((b1 - a1) * (b1 - c1));

    // Triangular transform — covariate 2
    if (x2[n] < a2 || x2[n] > b2)
      f2[n] = 0.0;
    else if (x2[n] <= c2)
      f2[n] = 2.0 * (x2[n] - a2) / ((b2 - a2) * (c2 - a2));
    else
      f2[n] = 2.0 * (b2 - x2[n]) / ((b2 - a2) * (b2 - c2));
  }
}

model {
  // --- Shape parameter priors (on log-gap scale) ---
  a1_raw      ~ normal(a1_star, sigma_a1);
  log_gap_c1  ~ normal(log(c1_star - a1_star), sigma_c1);
  log_gap_b1  ~ normal(log(b1_star - c1_star), sigma_b1);

  a2_raw      ~ normal(a2_star, sigma_a2);
  log_gap_c2  ~ normal(log(c2_star - a2_star), sigma_c2);
  log_gap_b2  ~ normal(log(b2_star - c2_star), sigma_b2);

  // --- Slope priors (expert direction + confidence) ---
  alpha  ~ normal(0, 1);
  beta1  ~ normal(0, 1);
  beta2  ~ normal(0, 1);

  // --- Likelihood ---
  y ~ bernoulli_logit(alpha + beta1 * f1 + beta2 * f2);
}

generated quantities {
  array[N] int y_rep;
  vector[N] log_lik;
  for (n in 1:N) {
    real mu = alpha + beta1 * f1[n] + beta2 * f2[n];
    y_rep[n]   = bernoulli_logit_rng(mu);
    log_lik[n] = bernoulli_logit_lpmf(y[n] | mu);
  }
}
