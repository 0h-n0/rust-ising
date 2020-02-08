extern crate ndarray;
extern crate ndarray_parallel;
extern crate rand;

#[macro_use]
extern crate failure;

use ndarray_parallel::prelude::*;
use rand::{distributions::Uniform, seq::SliceRandom, thread_rng, Rng};
use std::convert::TryFrom;
use std::convert::TryInto;

#[derive(Debug, Fail)]
enum Ising2DError {
    #[fail(display = "Simulator Error: {}", mesg)]
    Simulator { mesg: String },
    #[fail(display = "Plot Error: {}", mesg)]
    Plot { mesg: String },
}

struct Ising2D {
    width: usize,
    height: usize,
    inv_t: f64,
    J: f64, /* Coefficient of Interaction Eenrgy */
    total_E: f64,
    state: ndarray::Array2<i64>,
    energy_state: ndarray::Array2<f64>,
}

trait Plot: Sized {
    fn plot(self) -> Result<String, Ising2DError>;
}

trait Simulator: Sized {
    fn new(width: usize, height: usize, temperature: f64, J: f64) -> Self;
    fn run(&mut self, n_cycle: usize) -> Result<String, Ising2DError>;
    fn init_calc_energy(&mut self) -> f64;
}

fn boundary_check(w: i64, h: i64, width: usize, height: usize) -> (usize, usize) {
    let col = if w >= width as i64 {
        0
    } else if w < 0 {
        width - 1
    } else {
        w as usize
    };
    let row = if h >= height as i64 {
        0
    } else if h < 0 {
        height - 1
    } else {
        h as usize
    };
    (row, col)
}

impl Simulator for Ising2D {
    fn new(width: usize, height: usize, temperature: f64, J: f64) -> Self {
        let state = ndarray::Array1::<i64>::zeros(width * height)
            .map(move |x| x + [-1, 1].choose(&mut rand::thread_rng()).unwrap())
            .into_shape((width, height))
            .unwrap();
        let energy_state = ndarray::Array2::<f64>::zeros((width, height));
        let inv_t = 1. / temperature;
        let total_E = 0.;
        Ising2D {
            width,
            height,
            inv_t,
            J,
            total_E,
            state,
            energy_state,
        }
    }
    fn run(&mut self, n_cycle: usize) -> Result<String, Ising2DError> {
        self.total_E = self.init_calc_energy();
        println!("Initial state energy = {}", self.total_E);
        let mut rng = rand::thread_rng();
        let range_width: Vec<usize> = (0..self.width).collect();
        let range_height: Vec<usize> = (0..self.height).collect();
        println!("{:?}", range_width.choose(&mut rng).unwrap());
        for _ in 0..n_cycle {
            let i = *range_width.choose(&mut rng).unwrap();
            let j = *range_height.choose(&mut rng).unwrap();
            let dE = -self.energy_state[[i, j]];
            if (dE <= 0.) {
                self.state[[i, j]] = -self.state[[i, j]];
                self.energy_state[[i, j]] = -self.energy_state[[i, j]];
                self.total_E += dE;
            } else {
                let prob: f64 = rng.gen();
                if (prob < (-dE * self.inv_t).exp()) {
                    self.state[[i, j]] = -self.state[[i, j]];
                    self.energy_state[[i, j]] = -self.energy_state[[i, j]];
                    self.total_E += dE;
                }
            }
            println!("dE = {}, state energy = {}", dE, self.total_E)
        }
        Ok("Ok".to_string())
    }
    fn init_calc_energy(&mut self) -> f64 {
        let mut E = 0.;
        for i in 0..self.width {
            for j in 0..self.height {
                let ii: i64 = i.try_into().unwrap();
                let jj: i64 = j.try_into().unwrap();
                let top = boundary_check(ii, jj + 1, self.width, self.height);
                let bottom = boundary_check(ii, jj - 1, self.width, self.height);
                let right = boundary_check(ii + 1, jj, self.width, self.height);
                let left = boundary_check(ii - 1, jj, self.width, self.height);
                self.energy_state[[i, j]] +=
                    self.J * self.state[[i, j]] as f64 * self.state[top] as f64;
                self.energy_state[[i, j]] +=
                    self.J * self.state[[i, j]] as f64 * self.state[bottom] as f64;
                self.energy_state[[i, j]] +=
                    self.J * self.state[[i, j]] as f64 * self.state[right] as f64;
                self.energy_state[[i, j]] +=
                    self.J * self.state[[i, j]] as f64 * self.state[left] as f64;
            }
        }
        self.energy_state.sum()
    }
}

fn main() {
    let mut sim: Ising2D = Simulator::new(1024, 1024, 1., 1.);
    sim.run(100);
}
