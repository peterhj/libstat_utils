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

  pub fn get_mean_std(&self) -> (f64, f64) {
    (self.run_mean, (self.run_vars / (self.counter as f64 - 1.0)).sqrt())
  }
}
