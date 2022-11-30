//! As many classic numerical physics methods as I can remember.
#![deny(missing_docs)]

/// Helper module for handling data
pub mod data;

/// A crusty numerical integration module
pub mod integrate {
  extern crate nalgebra as na;
  use na::DVector;

  use crate::interpolate::chinterp;

  /// Fourth-order Runge-Kutta integrator.
  pub struct RK4<T: Fn(f64, &DVector<f64>) -> DVector<f64>> {
    f: T,
    h: f64,
  }

  impl<T: Fn(f64, &DVector<f64>) -> DVector<f64>> RK4<T> {
    /// Create a new RK4 integrator.
    pub fn new(f: T, h: f64) -> RK4<T> {
      RK4 { f, h }
    }

    /// Calculate `y(t + h)` given `y' = f(t, y), y(t) = y`.
    pub fn step(
      &self,
      t: f64,
      y: &DVector<f64>,
    ) -> Result<DVector<f64>, String> {
      let k1 = (self.f)(t, &y);
      let k2 = (self.f)(t + self.h / 2.0, &(y + self.h * &k1 / 2.0));
      let k3 = (self.f)(t + self.h / 2.0, &(y + self.h * &k2 / 2.0));
      let k4 = (self.f)(t + self.h, &(y + self.h * &k3));
      Ok(y + self.h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0)
    }

    /// Solve the IVP `y' = f(t, y), y(t0) = y0` for `y(t1)`. Note that this
    /// method does NOT perform dense output - the return value will be
    /// `y(t = t0 + nh)` such that `n = argmin_n(t1 - t0 - nh)`.
    ///
    /// ```
    /// use crusty::integrate::RK4;
    /// use assert_approx_eq::assert_approx_eq;
    /// use nalgebra::DVector;
    /// let rk4_integrator = RK4::new(|t, y: &DVector<f64>| { y.clone() }, 0.01);
    /// let y = rk4_integrator.ivp(0.0, 1.0, DVector::from_element(1, 1.0));
    /// assert_approx_eq!(y.unwrap()[(0)], 2.718281828459045, 1e-6);
    /// ```
    pub fn ivp(
      &self,
      t0: f64,
      t1: f64,
      y0: DVector<f64>,
    ) -> Result<DVector<f64>, String> {
      let mut y = y0;
      let mut t = t0;
      while t < t1 {
        y = self.step(t, &y).unwrap();
        t += self.h;
      }
      Ok(y)
    }

    /// Solve the IVP `y' = f(t, y), y(t0) = y0` for `y(t1)`. Supports
    /// sampling at dense output points using a Cubic Hermite spline.
    ///
    /// ```
    /// use crusty::integrate::RK4;
    /// use assert_approx_eq::assert_approx_eq;
    /// use nalgebra::DVector;
    /// let rk4_integrator = RK4::new(|t, y: &DVector<f64>| y.clone(), 0.01);
    /// let y = rk4_integrator.ivp_dense(0.0, 0.995, DVector::from_element(1, 1.0));
    /// assert_approx_eq!(y.unwrap()[(0)], 2.70472434128, 1e-6);
    /// ```
    pub fn ivp_dense(
      &self,
      t0: f64,
      t1: f64,
      y0: DVector<f64>,
    ) -> Result<DVector<f64>, String> {
      if t1 < t0 {
        return Err("t1 < t0".to_string());
      }

      let mut ya = y0.clone();
      let mut ta = t0;
      let mut yb = y0.clone();
      let mut tb = t0;
      while tb < t1 {
        ya = yb;
        ta = tb;
        yb = self.step(ta, &ya).unwrap();
        tb = self.h + ta;
      } // After this loop, [ya, yb] straddle the solution and ta < t1 < tb.

      // Construct Hermite interpolant and find value at t1
      let fa = (self.f)(ta, &ya);
      let fb = (self.f)(tb, &yb);

      Ok(chinterp(t1, ta, tb, &ya, &yb, &fa, &fb))
    }
  }

  /// Fourth(Fifth)-order Runge-Kutta-Fehlberg integrator, using an
  /// adaptive step size.
  pub struct RK45<T: Fn(f64, &DVector<f64>) -> DVector<f64>> {
    f: T,
    eps: f64,
  }

  impl<T: Fn(f64, &DVector<f64>) -> DVector<f64>> RK45<T> {
    /// Create a new RK45 integrator.
    pub fn new(f: T, eps: f64) -> RK45<T> {
      RK45 { f, eps }
    }

    ///
    pub fn step(
      &self,
      t: f64,
      y: &DVector<f64>,
      h: f64,
    ) -> Result<(DVector<f64>, f64), String> {
      let k1 = (self.f)(t, &y);
      let k2 = (self.f)(t + h / 4.0, &(y + h * &k1 / 4.0));
      let k3 = (self.f)(
        t + 3.0 * h / 8.0,
        &(y + h * (3.0 * &k1 / 32.0 + 9.0 * &k2 / 32.0)),
      );
      let k4 = (self.f)(
        t + 12.0 * h / 13.0,
        &(y
          + h
            * (1932.0 * &k1 / 2197.0 - 7200.0 * &k2 / 2197.0
              + 7296.0 * &k3 / 2197.0)),
      );
      let k5 = (self.f)(
        t + h,
        &(y
          + h
            * (439.0 * &k1 / 216.0 - 8.0 * &k2 + 3680.0 * &k3 / 513.0
              - 845.0 * &k4 / 4104.0)),
      );
      let k6 = (self.f)(
        t + h / 2.0,
        &(y
          + h
            * (-8.0 * &k1 / 27.0 + 2.0 * &k2 - 3544.0 * &k3 / 2565.0
              + 1859.0 * &k4 / 4104.0
              - 11.0 * &k5 / 40.0)),
      );
      let y1 = y
        + h
          * (25.0 * &k1 / 216.0
            + 1408.0 * &k3 / 2565.0
            + 2197.0 * &k4 / 4104.0
            - &k5 / 5.0);
      let y2 = y
        + h
          * (16.0 * &k1 / 135.0
            + 6656.0 * &k3 / 12825.0
            + 28561.0 * &k4 / 56430.0
            - 9.0 * &k5 / 50.0
            + 2.0 * &k6 / 55.0);
      let truncation_error = (y1.clone() - y2).norm();
      let hnew = 0.84 * h * (self.eps / truncation_error).powf(1.0 / 4.0);
      if truncation_error > self.eps {
        return self.step(t, y, hnew);
      } else {
        Ok((y1, h))
      }
    }

    /// Solve the IVP `y' = f(t, y), y(t0) = y0` for `y(t1)`. Supports
    /// sampling at dense output points using a Cubic Hermite spline.
    ///
    /// ```
    /// use crusty::integrate::RK45;
    /// use assert_approx_eq::assert_approx_eq;
    /// use nalgebra::DVector;
    /// let rk45_integrator = RK45::new(|t, y: &DVector<f64>| y.clone(), 1e-6);
    /// let y = rk45_integrator.ivp(0.0, 1.0, DVector::from_element(1, 1.0), 0.01);
    /// assert_approx_eq!(y.unwrap()[(0)], 2.718281828459045, 1e-6);
    /// ```
    pub fn ivp(
      &self,
      t0: f64,
      t1: f64,
      y0: DVector<f64>,
      h0: f64,
    ) -> Result<DVector<f64>, String> {
      if t1 < t0 {
        return Err("t1 < t0".to_string());
      }

      let mut ya = y0.clone();
      let mut ta = t0;
      let mut yb = y0.clone();
      let mut tb = t0;
      let mut h = h0;
      while tb < t1 {
        ya = yb;
        ta = tb;
        (yb, h) = self.step(ta, &ya, h).unwrap();
        tb = h + ta;
      } // After this loop, [ya, yb] straddle the solution and ta < t1 < tb.

      // Construct Hermite interpolant and find value at t1
      let fa = (self.f)(ta, &ya);
      let fb = (self.f)(tb, &yb);

      Ok(chinterp(t1, ta, tb, &ya, &yb, &fa, &fb))
    }
  }
}

/// A crusty numerical interpolation module
pub mod interpolate {
  use nalgebra::DVector;

  /// Return value of cubic Hermite spline interpolant between at `t` given
  /// (t0, y0, dy0) and (t1, y1, dy1).
  pub fn chinterp(
    t: f64,
    t0: f64,
    t1: f64,
    y0: &DVector<f64>,
    y1: &DVector<f64>,
    dy0: &DVector<f64>,
    dy1: &DVector<f64>,
  ) -> DVector<f64> {
    let s = (t - t0) / (t1 - t0);
    let h00 = 2.0 * s.powi(3) - 3.0 * s.powi(2) + 1.0;
    let h10 = s.powi(3) - 2.0 * s.powi(2) + s;
    let h01 = -2.0 * s.powi(3) + 3.0 * s.powi(2);
    let h11 = s.powi(3) - s.powi(2);

    h00 * y0 + h10 * (t1 - t0) * dy0 + h01 * y1 + h11 * (t1 - t0) * dy1
  }
}

/// A crusty Fast Fourier Transform module
pub mod fft {
  extern crate num_complex;
  use num_complex::Complex;

  /// Convert a vector of real numbers represented by f64 to a vector of
  /// the same numbers represented as Complex<f64>.
  ///
  /// This can be made faster using unsafe code but I am not confident enough
  /// to do so just yet, see
  /// https://stackoverflow.com/questions/54185667/how-to-safely-reinterpret-vecf64-as-vecnum-complexcomplexf64-with-half-t
  fn vec_real_to_complex(v: &Vec<f64>) -> Vec<Complex<f64>> {
    v.iter().map(|x| Complex::new(*x, 0.0)).collect()
  }

  /// Compute the discrete Fourier transform of `x` using the Danielson-Lanczos
  /// algorithm. Algorithm is carried out in-place.
  ///
  /// Algorithm and code obtained from Numerical Recipes in C, 3rd Edition.
  ///
  /// `isign = -1` for forward transform, `isign = 1` for inverse transform.
  ///
  /// ```
  /// use crusty::fft::fft;
  /// use assert_approx_eq::assert_approx_eq;
  /// use num_complex::Complex;
  ///
  /// let x = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0), Complex::new(4.0, 0.0)];
  /// let y = fft(x, -1).unwrap();
  /// println!("{:?}", y);
  /// assert_approx_eq!(y[0].re, 10.0, 1e-6);
  /// assert_approx_eq!(y[0].im, 0.0, 1e-6);
  /// assert_approx_eq!(y[1].re, -2.0, 1e-6);
  /// assert_approx_eq!(y[1].im, 2.0, 1e-6);
  /// assert_approx_eq!(y[2].re, -2.0, 1e-6);
  /// assert_approx_eq!(y[2].im, 0.0, 1e-6);
  /// assert_approx_eq!(y[3].re, -2.0, 1e-6);
  /// assert_approx_eq!(y[3].im, -2.0, 1e-6);
  /// ```
  pub fn fft(
    mut data: Vec<Complex<f64>>,
    isign: i32,
  ) -> Result<Vec<Complex<f64>>, String> {
    let n = data.len() as usize;
    if n < 2 || n & (n - 1) != 0 {
      return Err(String::from("data length is not a power of 2"));
    }

    let mut j = 0;
    for i in 0..n - 1 {
      if j > i {
        data.swap(i, j);
      }
      let mut m = n >> 1; // Set m to 1 in bit-reversed binary
      while m != 0 && j >= m {
        // From left to right in the binary representation of j, find the first
        // 0. On the way, set all 1s to 0s, and then set the first 0 to 1. This
        // procedure increments j by 1 in bit-reversed binary counting.
        j -= m;
        m >>= 1;
      }
      j += m; // set first 0 to 1
    }

    let mut m = 2;
    while n >= m {
      let theta = isign as f64 * (2.0 * std::f64::consts::PI / m as f64);
      let wtemp = (0.5 * theta).sin();
      let wm = Complex::new(-2.0 * wtemp * wtemp, (theta).sin());

      for k in (0..n).step_by(m) {
        let mut w = Complex::new(1.0, 0.0);
        for j in 0..m / 2 {
          let t = w * data[k + j + m / 2];
          let u = data[k + j];
          data[k + j] = u + t;
          data[k + j + m / 2] = u - t;
          w += wm * w;
        }
      }

      m <<= 1;
    }
    Ok(data)
  }

  /// Returns `crusty::fft:fft(x.clone())`
  ///
  /// ```
  /// use crusty::fft::fft_clone;
  /// use assert_approx_eq::assert_approx_eq;
  /// use num_complex::Complex;
  ///
  /// let x = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0), Complex::new(4.0, 0.0)];
  /// let y = fft_clone(&x, -1).unwrap();
  /// assert_approx_eq!(y[0].re, 10.0, 1e-6);
  /// assert_approx_eq!(y[0].im, 0.0, 1e-6);
  /// assert_approx_eq!(y[1].re, -2.0, 1e-6);
  /// assert_approx_eq!(y[1].im, 2.0, 1e-6);
  /// assert_approx_eq!(y[2].re, -2.0, 1e-6);
  /// assert_approx_eq!(y[2].im, 0.0, 1e-6);
  /// assert_approx_eq!(y[3].re, -2.0, 1e-6);
  /// assert_approx_eq!(y[3].im, -2.0, 1e-6);
  /// assert_approx_eq!(x[0].re, 1.0, 1e-6);
  /// ```
  pub fn fft_clone(
    x: &Vec<Complex<f64>>,
    isign: i32,
  ) -> Result<Vec<Complex<f64>>, String> {
    let y = x.clone();
    fft(y, isign)
  }

  /// Returns `crusty::fft:fft(y)`, where `y` is the Vector<Complex<f64>>
  /// representation of the Vector<f64> `x`.
  ///
  /// ```
  /// use crusty::fft::rfft_clone;
  /// use assert_approx_eq::assert_approx_eq;
  /// use num_complex::Complex;
  ///
  /// let x = vec![1.0, 2.0, 3.0, 4.0];
  /// let y = rfft_clone(&x, -1).unwrap();
  /// assert_approx_eq!(y[0].re, 10.0, 1e-6);
  /// assert_approx_eq!(y[0].im, 0.0, 1e-6);
  /// assert_approx_eq!(y[1].re, -2.0, 1e-6);
  /// assert_approx_eq!(y[1].im, 2.0, 1e-6);
  /// assert_approx_eq!(y[2].re, -2.0, 1e-6);
  /// assert_approx_eq!(y[2].im, 0.0, 1e-6);
  /// assert_approx_eq!(y[3].re, -2.0, 1e-6);
  /// assert_approx_eq!(y[3].im, -2.0, 1e-6);
  /// assert_approx_eq!(x[0], 1.0, 1e-6);
  /// ```
  pub fn rfft_clone(
    x: &Vec<f64>,
    isign: i32,
  ) -> Result<Vec<Complex<f64>>, String> {
    let y = vec_real_to_complex(x);
    fft(y, isign)
  }
}

#[cfg(test)]
mod tests {
  /// Tests `crusty::fft::fft` against precomputed fft of a random array
  /// using Python+numpy
  #[test]
  fn test_fft_big_vec() {
    use super::fft::rfft_clone;
    use assert_approx_eq::assert_approx_eq;

    let x = vec![
      0.36117797, 0.75825683, 0.00786384, 0.16040471, 0.40329213, 0.22808298,
      0.32710224, 0.8247796, 0.85168761, 0.05081063, 0.63342277, 0.10148653,
      0.82063687, 0.12473368, 0.3725931, 0.28827866,
    ];

    let y = rfft_clone(&x, -1).unwrap();
    assert_approx_eq!(y[0].re, 6.31461013, 1e-6);
    assert_approx_eq!(y[1].re, -0.75974994, 1e-6);
    assert_approx_eq!(y[2].re, 0.91342059, 1e-6);
    assert_approx_eq!(y[3].re, 0.02612736, 1e-6);
    assert_approx_eq!(y[4].re, 1.09581263, 1e-6);
    assert_approx_eq!(y[5].re, -0.18680651, 1e-6);
    assert_approx_eq!(y[6].re, -0.93554742, 1e-6);
    assert_approx_eq!(y[7].re, -1.04160947, 1e-6);
    assert_approx_eq!(y[8].re, 1.24094292, 1e-6);
    assert_approx_eq!(y[9].re, -1.04160947, 1e-6);
    assert_approx_eq!(y[10].re, -0.93554742, 1e-6);
    assert_approx_eq!(y[11].re, -0.18680651, 1e-6);
    assert_approx_eq!(y[12].re, 1.09581263, 1e-6);
    assert_approx_eq!(y[13].re, 0.02612736, 1e-6);
    assert_approx_eq!(y[14].re, 0.91342059, 1e-6);
    assert_approx_eq!(y[15].re, -0.75974994, 1e-6);

    assert_approx_eq!(y[0].im, 0., 1e-6);
    assert_approx_eq!(y[1].im, 0.26589503, 1e-6);
    assert_approx_eq!(y[2].im, 0.33765665, 1e-6);
    assert_approx_eq!(y[3].im, -1.03000111, 1e-6);
    assert_approx_eq!(y[4].im, 0.21306537, 1e-6);
    assert_approx_eq!(y[5].im, -1.14431934, 1e-6);
    assert_approx_eq!(y[6].im, 0.22083921, 1e-6);
    assert_approx_eq!(y[7].im, -1.51780216, 1e-6);
    assert_approx_eq!(y[8].im, 0., 1e-6);
    assert_approx_eq!(y[9].im, 1.51780216, 1e-6);
    assert_approx_eq!(y[10].im, -0.22083921, 1e-6);
    assert_approx_eq!(y[11].im, 1.14431934, 1e-6);
    assert_approx_eq!(y[12].im, -0.21306537, 1e-6);
    assert_approx_eq!(y[13].im, 1.03000111, 1e-6);
    assert_approx_eq!(y[14].im, -0.33765665, 1e-6);
    assert_approx_eq!(y[15].im, -0.26589503, 1e-6);
  }

  /// Tests `crusty::fft::fft` against a purely imaginary input
  #[test]
  fn test_fft_pure_imag() {
    use super::fft::fft;
    use assert_approx_eq::assert_approx_eq;
    use num_complex::Complex;
    let x = vec![
      Complex::new(0.0, 1.0),
      Complex::new(0.0, 2.0),
      Complex::new(0.0, 3.0),
      Complex::new(0.0, 4.0),
    ];
    let y = fft(x, -1).unwrap();
    println!("{:?}", y);
    assert_approx_eq!(y[0].re, 0.0, 1e-6);
    assert_approx_eq!(y[0].im, 10.0, 1e-6);
    assert_approx_eq!(y[1].re, -2.0, 1e-6);
    assert_approx_eq!(y[1].im, -2.0, 1e-6);
    assert_approx_eq!(y[2].re, 0.0, 1e-6);
    assert_approx_eq!(y[2].im, -2.0, 1e-6);
    assert_approx_eq!(y[3].re, 2.0, 1e-6);
    assert_approx_eq!(y[3].im, -2.0, 1e-6);
  }
}
