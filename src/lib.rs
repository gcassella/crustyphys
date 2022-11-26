//! As many classic numerical physics methods as I can remember.
#![deny(missing_docs)]
#![deny(rustdoc::missing_doc_code_examples)]

/// A crusty numerical integration library
pub mod integrate {
    extern crate nalgebra as na;
    use na::DVector;

    /// Fourth-order Runge-Kutta integrator.
    pub struct RK4 {
        f: fn(f64, &DVector<f64>) -> DVector<f64>,
        h: f64,
    }

    impl RK4 {
        /// Create a new RK4 integrator.
        pub fn new(f: fn(f64, &DVector<f64>) -> DVector<f64>, h: f64) -> RK4 {
            RK4 { f, h }
        }

        /// Calculate `y(t + h)` given `y' = f(t, y), y(t) = y`.
        pub fn step(&self, t: f64, y: &DVector<f64>) -> Result<DVector<f64>, String> {
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
        pub fn ivp(&self, t0: f64, t1: f64, y0: DVector<f64>) -> Result<DVector<f64>, String> {
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
                println!("{} {} {} {}", ta, ya[(0)], tb, yb[(0)]);
            } // After this loop, [ya, yb] straddle the solution and ta < t1 < tb.

            // Construct Hermite interpolant and find value at t1
            let fa = (self.f)(ta, &ya);
            let fb = (self.f)(tb, &yb);

            let s = (t1 - ta) / (tb - ta);
            let h00 = 2.0 * s.powi(3) - 3.0 * s.powi(2) + 1.0;
            let h10 = s.powi(3) - 2.0 * s.powi(2) + s;
            let h01 = -2.0 * s.powi(3) + 3.0 * s.powi(2);
            let h11 = s.powi(3) - s.powi(2);

            Ok(h00 * ya + h10 * (tb - ta) * fa + h01 * yb + h11 * (tb - ta) * fb)
        }
    }

    /// Fourth(Fifth)-order Runge-Kutta-Fehlberg integrator, using an
    /// adaptive step size.
    pub struct RK45 {
        f: fn(f64, &DVector<f64>) -> DVector<f64>,
        eps: f64,
    }

    impl RK45 {
        /// Create a new RK45 integrator.
        pub fn new(f: fn(f64, &DVector<f64>) -> DVector<f64>, eps: f64) -> RK45 {
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
                &(y + h * (1932.0 * &k1 / 2197.0 - 7200.0 * &k2 / 2197.0 + 7296.0 * &k3 / 2197.0)),
            );
            let k5 = (self.f)(
                t + h,
                &(y + h * (439.0 * &k1 / 216.0 - 8.0 * &k2 + 3680.0 * &k3 / 513.0 - 845.0 * &k4 / 4104.0)),
            );
            let k6 = (self.f)(
                t + h / 2.0,
                &(y + h
                    * (-8.0 * &k1 / 27.0 + 2.0 * &k2 - 3544.0 * &k3 / 2565.0 + 1859.0 * &k4 / 4104.0
                        - 11.0 * &k5 / 40.0)),
            );
            let y1 = y + h
                * (25.0 * &k1 / 216.0 + 1408.0 * &k3 / 2565.0 + 2197.0 * &k4 / 4104.0 - &k5 / 5.0);
            let y2 = y + h
                * (16.0 * &k1 / 135.0 + 6656.0 * &k3 / 12825.0 + 28561.0 * &k4 / 56430.0
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

        /// Solve the IVP `y' = f(t, y), y(t0) = y0` for `y(t1)`. Note that this
        /// method does NOT perform dense output - the return value will be
        /// `y(t = t0 + nh)` such that `n = argmin_n(t1 - t0 - nh)`.
        ///
        /// ```
        /// use crusty::integrate::RK45;
        /// use assert_approx_eq::assert_approx_eq;
        /// use nalgebra::DVector;
        /// let rk45_integrator = RK45::new(|t, y: &DVector<f64>| y.clone(), 1e-6);
        /// let y = rk45_integrator.ivp(0.0, 1.0, DVector::from_element(1, 1.0), 0.01);
        /// assert_approx_eq!(y.unwrap()[(0)], 2.718281828459045, 1e-6);
        /// ```
        pub fn ivp(&self, t0: f64, t1: f64, y0: DVector<f64>, h0: f64) -> Result<DVector<f64>, String> {
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

            let s = (t1 - ta) / (tb - ta);
            let h00 = 2.0 * s.powi(3) - 3.0 * s.powi(2) + 1.0;
            let h10 = s.powi(3) - 2.0 * s.powi(2) + s;
            let h01 = -2.0 * s.powi(3) + 3.0 * s.powi(2);
            let h11 = s.powi(3) - s.powi(2);

            Ok(h00 * ya + h10 * (tb - ta) * fa + h01 * yb + h11 * (tb - ta) * fb)
        }
    }
}

#[cfg(test)]
mod tests {}
