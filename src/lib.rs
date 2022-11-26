//! As many classic numerical physics methods as I can remember.
#![deny(missing_docs)]
#![deny(rustdoc::missing_doc_code_examples)]

/// A crusty numerical physics library.
pub mod integrate {
    /// Fourth-order Runge-Kutta integrator.
    pub struct RK4 {
        f: fn(f64, f64) -> f64,
        h: f64
    }

    impl RK4 {
        /// Create a new RK4 integrator.
        pub fn new(f: fn(f64, f64) -> f64, h: f64) -> RK4 {
            RK4 { f, h }
        }

        /// Calculate `y(t + h)` given `y' = f(t, y), y(t) = y`.
        pub fn step(&self, t: f64, y: f64) -> Result<f64, String> {
            let k1 = (self.f)(t, y);
            let k2 = (self.f)(t + self.h / 2.0, y + self.h * k1 / 2.0);
            let k3 = (self.f)(t + self.h / 2.0, y + self.h * k2 / 2.0);
            let k4 = (self.f)(t + self.h, y + self.h * k3);
            Ok(y + self.h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0)
        }

        /// Solve the IVP `y' = f(t, y), y(t0) = y0` for `y(t1)`. Note that this
        /// method does NOT perform dense output - the return value will be
        /// `y(t = t0 + nh)` such that `n = argmin_n(t1 - t0 - nh)`.
        ///
        /// ```
        /// use crusty::integrate::RK4;
        /// use assert_approx_eq::assert_approx_eq;
        /// let rk4_integrator = RK4::new(|t, y| y, 0.01);
        /// let y = rk4_integrator.ivp(0.0, 1.0, 1.0);
        /// assert_approx_eq!(y.unwrap(), 2.718281828459045, 1e-6);
        /// ```
        pub fn ivp(&self, t0: f64, t1: f64, y0: f64) -> Result<f64, String> {
            let mut y = y0;
            let mut t = t0;
            while t < t1 {
                y = self.step(t, y).unwrap();
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
        /// let rk4_integrator = RK4::new(|t, y| y, 0.01);
        /// let y = rk4_integrator.ivp_dense(0.0, 0.995, 1.0);
        /// assert_approx_eq!(y.unwrap(), 2.70472434128, 1e-6);
        /// ```
        pub fn ivp_dense(&self, t0: f64, t1: f64, y0: f64) -> Result<f64, String> {
            let mut ya = y0;
            let mut ta = t0;
            let mut yb = y0;
            let mut tb = t0;
            while tb < t1 {
                ya = yb;
                ta = tb;
                yb = self.step(ta, ya).unwrap();
                tb = self.h + ta;
            } // After this loop, [ya, yb] straddle the solution and ta < t1 < tb.

            // Construct Hermite interpolant and find value at t1
            let fa = (self.f)(ta, ya);
            let fb = (self.f)(tb, yb);

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
