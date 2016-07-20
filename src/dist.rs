use float::ord::{F64InfNan};
//use gsl::{Gsl};
use gsl::functions::{erf};
use gsl::integration::{IntegrationWorkspace, Integrand};

use libc::{c_void};
use std::f64::consts::{PI};
use std::cell::{RefCell};
use std::collections::{BTreeMap, Bound};
use std::iter::{repeat};
use std::mem::{transmute};
use std::rc::{Rc};

pub trait Dist {
  fn cdf(&self, z: f64) -> f64;
}

pub trait Density {
  fn pdf(&self, z: f64) -> f64;

  fn support(&self) -> (Option<f64>, Option<f64>) {
    (None, None)
  }
}

pub trait EmpiricalDensityEstimator: Density {
}

#[derive(Clone)]
pub struct EmpiricalDist<DensityEst> where DensityEst: EmpiricalDensityEstimator {
  dist: EmpiricalCdf,
  hist: DensityEst,
}

impl EmpiricalDist<HistogramDensityEstimator> {
  pub fn new(hist_step: f64, xs: &mut [f64]) -> EmpiricalDist<HistogramDensityEstimator> {
    xs.sort_by_key(|&x| F64InfNan(x));
    // FIXME(201607xx): skip xs until we find non-nan.
    let min_x = xs[0];
    let max_x = xs[xs.len()-1];
    let dist = EmpiricalCdf::new(xs);
    let hist = HistogramDensityEstimator::new(min_x, max_x, hist_step, xs);
    EmpiricalDist{
      dist: dist,
      hist: hist,
    }
  }
}

impl EmpiricalDist<KernelDensityEstimator> {
  pub fn new(xs: &mut [f64]) -> EmpiricalDist<KernelDensityEstimator> {
    unimplemented!();
  }
}

impl<DensityEst> Dist for EmpiricalDist<DensityEst> where DensityEst: EmpiricalDensityEstimator {
  fn cdf(&self, z: f64) -> f64 {
    self.dist.cdf(z)
  }
}

impl<DensityEst> Density for EmpiricalDist<DensityEst> where DensityEst: EmpiricalDensityEstimator {
  fn pdf(&self, z: f64) -> f64 {
    self.hist.pdf(z)
  }

  fn support(&self) -> (Option<f64>, Option<f64>) {
    self.hist.support()
  }
}

#[derive(Clone)]
pub struct EmpiricalCdf {
  rank: BTreeMap<F64InfNan, f64>,
}

impl EmpiricalCdf {
  pub fn new(sorted_xs: &[f64]) -> EmpiricalCdf {
    let n = sorted_xs.len() as f64;
    let mut rank = BTreeMap::new();
    for (i, &x) in sorted_xs.iter().enumerate() {
      assert!(!x.is_nan());
      let y = (i+1) as f64 / n;
      rank.insert(F64InfNan(-x), y);
    }
    EmpiricalCdf{
      rank: rank,
    }
  }
}

impl Dist for EmpiricalCdf {
  fn cdf(&self, z: f64) -> f64 {
    for (_, &y) in self.rank.range(Bound::Included(&F64InfNan(-z)), Bound::Unbounded) {
      return y;
    }
    0.0
  }
}

#[derive(Clone)]
pub struct HistogramDensityEstimator {
  hist: Vec<i32>,
  lo:   f64,
  hi:   f64,
  step: f64,
  num_pts:  usize,
  num_bins: usize,
  norm: f64,
}

impl HistogramDensityEstimator {
  pub fn new(min: f64, max: f64, step: f64, xs: &[f64]) -> HistogramDensityEstimator {
    assert!(step > 0.0);
    let num_pts = xs.len();
    let num_bins = ((max / step).ceil() as isize - (min / step).floor() as isize + 1) as usize;
    let offset = (min / step).floor() as isize;
    let lo = (min / step).floor() * step;
    let hi = (max / step).ceil() * step;
    //println!("DEBUG: hist: bins: {} off: {} lo: {} hi: {}", num_bins, offset, lo, hi);
    let mut hist: Vec<i32> = repeat(0).take(num_bins).collect();
    for &x in xs.iter() {
      let bin = (x / step).ceil() as isize - offset;
      assert!(bin >= 0);
      let i = bin as usize;
      hist[i] += 1;
    }
    /*for i in 0 .. num_bins {
      println!("DEBUG: hist: bin[{}]: {}", i, hist[i]);
    }*/
    HistogramDensityEstimator{
      hist: hist,
      lo:   lo,
      hi:   hi,
      step: step,
      num_pts:  num_pts,
      num_bins: num_bins,
      norm: 1.0 / xs.len() as f64,
    }
  }
}

impl Density for HistogramDensityEstimator {
  fn pdf(&self, z: f64) -> f64 {
    if z < self.lo {
      //println!("DEBUG: hist: z: {} y: {}", z, 0.0);
      0.0
    } else if z > self.hi {
      //println!("DEBUG: hist: z: {} y: {}", z, 0.0);
      0.0
    } else {
      let i = ((z - self.lo) / self.step).floor() as usize;
      let y = self.hist[i] as f64 / (self.step * self.num_pts as f64);
      y
    }
  }

  fn support(&self) -> (Option<f64>, Option<f64>) {
    (Some(self.lo), Some(self.hi))
  }
}

impl EmpiricalDensityEstimator for HistogramDensityEstimator {
}

pub struct KernelDensityEstimator {}

impl Density for KernelDensityEstimator {
  fn pdf(&self, z: f64) -> f64 {
    unimplemented!();
  }

  fn support(&self) -> (Option<f64>, Option<f64>) {
    unimplemented!();
  }
}

impl EmpiricalDensityEstimator for KernelDensityEstimator {
}

#[derive(Clone)]
pub struct GaussianDist {
  pub mean: f64,
  pub std:  f64,
}

impl Dist for GaussianDist {
  fn cdf(&self, z: f64) -> f64 {
    let t = (z - self.mean) / self.std.abs();
    0.5 + 0.5 * erf(t / 2.0f64.sqrt())
  }
}

impl Density for GaussianDist {
  fn pdf(&self, z: f64) -> f64 {
    let t = (z - self.mean) / self.std.abs();
    1.0 / ((2.0 * PI).sqrt() * self.std.abs()) * (-0.5 * t * t).exp()
  }
}

pub struct ProductDist<D> where D: Dist {
  dists:    Vec<D>,
  scratch:  RefCell<Vec<f64>>,
}

impl<D> ProductDist<D> where D: Dist {
  pub fn new(dists: Vec<D>) -> ProductDist<D> {
    let num_dists = dists.len();
    ProductDist{
      dists:    dists,
      scratch:  RefCell::new(repeat(0.0).take(num_dists).collect()),
    }
  }
}

impl<D> Dist for ProductDist<D> where D: Dist {
  fn cdf(&self, z: f64) -> f64 {
    let mut y = 1.0;
    for dist in self.dists.iter() {
      y *= dist.cdf(z);
    }
    y
  }
}

impl<D> Density for ProductDist<D> where D: Dist + Density {
  fn pdf(&self, z: f64) -> f64 {
    let num_dists = self.dists.len();
    let mut scratch = self.scratch.borrow_mut();
    for j in 0 .. num_dists {
      scratch[j] = self.dists[j].cdf(z);
    }
    let mut y = 0.0;
    for j in 0 .. num_dists {
      let mut yy = 1.0;
      for k in 0 .. num_dists {
        if j == k {
          yy *= self.dists[k].pdf(z);
        } else {
          yy *= scratch[k];
        }
      }
      y += yy;
    }
    y
  }

  fn support(&self) -> (Option<f64>, Option<f64>) {
    let mut lo = None;
    let mut hi = None;
    for dist in self.dists.iter() {
      let (dlo, dhi) = dist.support();
      if let Some(dlo) = dlo {
        match lo {
          None          => lo = Some(dlo),
          Some(prev_lo) => lo = Some(prev_lo.min(dlo)),
        }
      }
      if let Some(dhi) = dhi {
        match hi {
          None          => hi = Some(dhi),
          Some(prev_hi) => hi = Some(prev_hi.max(dhi)),
        }
      }
    }
    (lo, hi)
  }
}

pub struct Expectation<D> where D: Density {
  density:  Rc<D>,
}

impl<D> Expectation<D> where D: Density {
  pub fn new(density: Rc<D>) -> Expectation<D> {
    Expectation{density: density}
  }

  extern "C" fn mean_integrand(x: f64, data: *mut c_void) -> f64 {
    let dist: &Rc<D> = unsafe { transmute(data) };
    let dy = x * dist.pdf(x);
    dy
  }

  pub fn mean(&self) -> f64 {
    let num_iters = 10000;
    let mut workspace = IntegrationWorkspace::new(num_iters);
    let mut integrand = Integrand{
      function: Expectation::<D>::mean_integrand,
      data:     self.density.clone(),
    };
    match self.density.support() {
      (None, None) => {
        let (val, _) = integrand.integrate_qagi(0.0, 1.0e-4, num_iters, &mut workspace);
        val
      }
      (Some(a), Some(b)) => {
        let (val, _) = integrand.integrate_qags(a, b, 0.0, 1.0e-4, num_iters, &mut workspace);
        val
      }
      _ => unimplemented!(),
    }
  }
}

#[test]
fn test_gaussian_mean() {
  //Gsl::disable_error_handler();
  let exp = Expectation::new(Rc::new(GaussianDist{mean: 1.0, std: 2.0}));
  println!("TEST: mean: {}", exp.mean());
  panic!();
}

#[test]
fn test_max_1x_gaussian_mean() {
  //Gsl::disable_error_handler();
  let exp = Expectation::new(Rc::new(ProductDist::new(vec![GaussianDist{mean: 0.0, std: 1.0}])));
  println!("TEST: mean (max 1x): {}", exp.mean());
  panic!();
}

#[test]
fn test_max_2x_gaussian_mean() {
  //Gsl::disable_error_handler();
  let exp = Expectation::new(Rc::new(ProductDist::new(vec![GaussianDist{mean: 0.0, std: 1.0}, GaussianDist{mean: 0.0, std: 1.0}])));
  println!("TEST: mean (max 2x): {}", exp.mean());
  panic!();
}

#[test]
fn test_max_4x_gaussian_mean() {
  //Gsl::disable_error_handler();
  let exp = Expectation::new(Rc::new(ProductDist::new(vec![GaussianDist{mean: 0.0, std: 1.0}, GaussianDist{mean: 0.0, std: 1.0}, GaussianDist{mean: 0.0, std: 1.0}, GaussianDist{mean: 0.0, std: 1.0}])));
  //let exp = Expectation::new(Rc::new(ProductDist::new(vec![GaussianDist{mean: 0.307, std: 0.010}, GaussianDist{mean: 0.307, std: 0.010}, GaussianDist{mean: 0.307, std: 0.010}, GaussianDist{mean: 0.307, std: 0.010}])));
  println!("TEST: mean (max 4x): {}", exp.mean());
  panic!();
}
