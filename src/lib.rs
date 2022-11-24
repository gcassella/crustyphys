//! As many classic numerical physics methods as I can remember.
#![deny(missing_docs)]
#![deny(rustdoc::missing_doc_code_examples)]

/// A crusty numerical physics library.
pub mod crusty {
    /// Given f(t, y) = dy/dt return a function f(t, y, h) that
    /// returns the value of y at t+h using the fourth order Runge-Kutta method.
    ///
    /// ```
    /// use crusty::crusty::rk4factory;
    /// use assert_approx_eq::assert_approx_eq;
    /// let h = 0.01;
    /// let mut t = 0.0;
    /// let mut y = 1.0;
    /// let rk4step = rk4factory(|_t, y| y);
    /// for _ in 0..100 {
    ///     y = rk4step(t, y, h);
    ///     t = t + h;
    ///     println!("t = {}, y = {}", t, y);
    /// }
    /// assert_approx_eq!(y, 2.718281828459045, 1e-6);
    /// ```
    pub fn rk4factory(f: impl Fn(f64, f64) -> f64) -> impl Fn(f64, f64, f64) -> f64 {
        return move |t, y, h| {
            let k1 = f(t, y);
            let k2 = f(t + h / 2.0, y + h * k1 / 2.0);
            let k3 = f(t + h / 2.0, y + h * k2 / 2.0);
            let k4 = f(t + h, y + h * k3);

            y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        };
    }
}

#[cfg(test)]
mod tests {}
