#[derive(Clone)]
pub struct OnlineMeanVar {
  counter:  usize,
  run_mean: f64,
  run_vars: f64,
}

impl OnlineMeanVar {
  pub fn new() -> OnlineMeanVar {
    OnlineMeanVar{
      counter:  0,
      run_mean: 0.0,
      run_vars: 0.0,
    }
  }

  pub fn reset(&mut self) {
    self.counter = 0;
    self.run_mean = 0.0;
    self.run_vars = 0.0;
  }

  pub fn update(&mut self, x: f64) {
    let prev_mean = self.run_mean;
    self.counter += 1;
    let n = self.counter as f64;
    self.run_mean += (x - prev_mean) / n;
    self.run_vars += (x - prev_mean) * (x - self.run_mean);
  }

  pub fn get_mean(&self) -> f64 {
    self.run_mean
  }

  pub fn get_mean_std(&self) -> (f64, f64) {
    (self.run_mean, (self.run_vars / (self.counter as f64 - 1.0)).sqrt())
  }
}

#[derive(Clone)]
pub struct OnlineAutocorr1 {
  counter:  usize,
  lag0_est: OnlineMeanVar,
  lag1_est: OnlineMeanVar,
  lag1_val: f64,
  corr1s:   f64,
}

impl OnlineAutocorr1 {
  pub fn new() -> OnlineAutocorr1 {
    OnlineAutocorr1{
      counter:  0,
      lag0_est: OnlineMeanVar::new(),
      lag1_est: OnlineMeanVar::new(),
      lag1_val: 0.0,
      corr1s:   0.0,
    }
  }

  pub fn reset(&mut self) {
    self.counter = 0;
    self.lag0_est.reset();
    self.lag1_est.reset();
    self.lag1_val = 0.0;
    self.corr1s = 0.0;
  }

  pub fn update(&mut self, x: f64) {
    if self.counter > 0 {
      let prev_lag0_mean = self.lag0_est.get_mean();
      self.lag0_est.update(x);
      let lag0_mean = self.lag0_est.get_mean();
      let prev_lag1_mean = self.lag1_est.get_mean();
      self.lag1_est.update(self.lag1_val);
      let lag1_mean = self.lag1_est.get_mean();
      let n = self.lag1_est.counter as f64;
      self.corr1s += (x - lag0_mean) * (self.lag1_val - lag1_mean) + (n - 1.0) / (n * n) * (x - prev_lag0_mean) * (self.lag1_val - prev_lag1_mean);
    }
    self.counter += 1;
    self.lag1_val = x;
  }

  pub fn get_autocorr1(&self) -> f64 {
    let (_, lag0_std) = self.lag0_est.get_mean_std();
    let (_, lag1_std) = self.lag1_est.get_mean_std();
    self.corr1s / (self.lag1_est.counter as f64 * lag0_std * lag1_std)
  }
}
