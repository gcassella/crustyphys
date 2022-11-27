extern crate nalgebra as na;
use crusty::integrate::RK45;
use na::DVector;

use std::{fs::File, io::Write};

/// Given phase space coordinates ({x_i}, {v_i}) of a set of masses {m_i},
/// calculate the gravitational force f_i acting on each mass.
fn force(y: &DVector<f64>, masses: &Vec<f64>) -> DVector<f64> {
  let mut f = DVector::from_element(y.len(), 0.0);
  let n = y.len() / 2 as usize;
  for i in 0..n {
    for j in i..n {
      if i != j {
        let r = y.slice((2 * i, 0), (2, 1)) - y.slice((2 * j, 0), (2, 1));
        let r2 = r.norm_squared();
        let r3 = r2 * r.norm();

        f[(2 * i)] -= masses[j] * r[(0)] / r3;
        f[(2 * i + 1)] -= masses[j] * r[(1)] / r3;
        f[(2 * j)] += masses[i] * r[(0)] / r3;
        f[(2 * j + 1)] += masses[i] * r[(1)] / r3;
      }
    }
  }
  f
}

/// Vertically concatenate contents of column vectors u and v.
fn cat_vec(u: &DVector<f64>, v: &DVector<f64>) -> DVector<f64> {
  let mut w = DVector::from_element(u.len() + v.len(), 0.0);
  w.slice_mut((0, 0), (u.len(), 1)).copy_from(u);
  w.slice_mut((u.len(), 0), (v.len(), 1)).copy_from(v);
  w
}

fn make_ode_sys(
  masses: Vec<f64>,
) -> impl Fn(f64, &DVector<f64>) -> DVector<f64> {
  move |_t: f64, y: &DVector<f64>| {
    let num_bodies = y.len() / 2 as usize;
    let x: DVector<f64> = y.rows(0, num_bodies).clone_owned();
    let v: DVector<f64> = y.rows(num_bodies, num_bodies).clone_owned();
    let curr_force = force(&(x), &masses);
    cat_vec(&v, &curr_force)
  }
}

/// Write the time `t` and phase space positions `y` to a CSV file.
fn write_out(file: &mut File, t: f64, y: &DVector<f64>) -> () {
  file
    .write(format!("{t}").as_bytes())
    .expect("Unable to write to file");
  for i in 0..y.len() {
    file.write(b",").expect("Unable to write to file");
    let val = y[(i)];
    file
      .write(format!("{val}").as_bytes())
      .expect("Unable to write to file");
  }
  file
    .write("\n".as_bytes())
    .expect("Unable to write to file");
}

fn main() -> () {
  // Define the position and velocity of three initial masses.
  // With only three bodies, our dynamics exists on a plane so we can work in
  // 2D.
  //
  // These initial values produce a somewhat interesting orbit, with a heavy
  // planet in orbit around a star, and a small satellite orbitting the planet.

  let masses = vec![1e12, 1.0, 0.001];

  let x0 = DVector::from_vec(vec![0.0, 0.0, 10.0, 0.0, 10.01, 0.0]);

  let v0 = DVector::from_vec(vec![0.0, 0.0, 0.0, 1e5, 0.0, 1e5 + 2e4]);

  let mut y0 = cat_vec(&x0, &v0);

  // Full enumeration of the system of ODEs:
  //
  // dx0_0/dt = v0_0
  // dx0_1/dt = v0_1
  // dx1_0/dt = v1_0
  // dx1_1/dt = v1_1
  // dx2_0/dt = v2_0
  // dx2_1/dt = v2_1
  // dv0_0/dt = -(normed(r01)[0] / r01^2 + normed(r01)[0] / r02^2)
  // dv0_1/dt = -(normed(r01)[1] / r01^2 + normed(r01)[1] / r02^2)
  // dv1_0/dt = -(normed(r10)[0] / r10^2 + normed(r12)[0] / r12^2)
  // dv1_1/dt = -(normed(r10)[1] / r10^2 + normed(r12)[1] / r12^2)
  // dv2_0/dt = -(normed(r20)[0] / r20^2 + normed(r21)[0] / r21^2)
  // dv2_1/dt = -(normed(r20)[1] / r20^2 + normed(r21)[1] / r21^2)
  //
  // where rij = xi - xj

  println!("Initial state: {y0}", y0 = y0);
  let mut t0 = 0.0;
  let mut h0 = 0.01;
  // Integrate the system of ODEs using the Runge-Kutta-Fehlberg 4/5 method
  let ode_sys = make_ode_sys(masses);
  let solver = RK45::new(ode_sys, 1e-5);

  let mut file = File::create("three_body.csv").expect("Unable to create file");

  file
    .write(b"t,x0,y0,x1,y1,x2,y2,vx0,vy0,vx1,vy1,vx2,vy2")
    .expect("Unable to write to file");

  for _ in 0..1000000 {
    write_out(&mut file, t0, &y0);
    let (y1, h1) = solver.step(t0, &y0, h0).unwrap();
    let t1 = t0 + h0;

    y0 = y1;
    t0 = t1;
    h0 = h1;
  }
}
