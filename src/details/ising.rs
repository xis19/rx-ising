extern crate ndarray;

use ndarray::{prelude::*, IntoDimension};
use rand::prelude::*;

use super::sample::Sample;
use super::types::Energy;

pub type Ix = ndarray::Ix;

pub struct Ising<const D: usize, const PBC: bool> {
    b: Energy,
    // XXX: At this stage we only consider the nearest neighbors. To introduce "long-range" interactions j has to be
    // a function of distance.
    j: Energy,
    model: Array<i8, Dim<[Ix; D]>>,
    rng: ThreadRng,
}

impl<const D: usize, const PBC: bool> Ising<D, PBC>
where
    [Ix; D]: IntoDimension<Dim = Dim<[Ix; D]>>,
    Dim<[Ix; D]>: Dimension,
{
    pub fn new(b: Energy, j: Energy, d: [Ix; D]) -> Self {
        Self {
            b,
            j,
            model: Array::<i8, Dim<[Ix; D]>>::zeros(d),
            rng: thread_rng(),
        }
    }

    pub fn get_b(&self) -> Energy {
        self.b
    }

    pub fn get_j(&self) -> Energy {
        self.j
    }

    // Sum of the spin over the whole model
    // NOTE: It is not possible to use self.model.sum() since the type is i8, and might trigger overflow for large systems.
    fn model_sum(&self) -> i32 {
        self.model.iter().map(|i| *i as i32).fold(0, |s, v| s + v)
    }

    fn get_neighbor(&self, pos: &[Ix; D], index: usize) -> Option<[Ix; D]> {
        let max = self.model.shape()[index];
        let mut result = *pos;
        result[index] += 1;

        if !PBC {
            if result[index] >= max {
                None
            } else {
                Some(result)
            }
        } else {
            if max < 2 {
                // This axis is flat, no neighbor
                None
            } else {
                if result[index] >= max {
                    result[index] = 0;
                }
                Some(result)
            }
        }
    }

    // Evalutes the total energy of the system
    pub fn get_energy(&self) -> Energy {
        let mut energy: Energy = 0.0;

        // Sum up the external field term
        energy += -self.b * (self.model_sum() as f64);

        // Sum up all the pairwise terms
        let mut pariwise_energy = 0.0;
        for i in self.model.indexed_iter() {
            let position: [Ix; D] =
                i.0.into_dimension()
                    .as_array_view()
                    .to_slice()
                    .expect("Must have dimension")
                    .try_into()
                    .expect("Must be convertable to [Ix; D]");
            let value = i.1;
            let mut s = 0;
            for index in 0..D {
                let option_neighbor = self.get_neighbor(&position, index);
                match option_neighbor {
                    None => {
                        continue;
                    }
                    Some(neighbor) => {
                        s += self.model[neighbor.into_dimension()];
                    }
                }
            }
            pariwise_energy += (*value as f64) * (s as f64);
        }
        energy += -self.j * pariwise_energy;

        energy
    }

    // Order parameter of the current model
    pub fn get_order_parameter(&self) -> f64 {
        (self.model_sum() as f64) / (self.model.len() as f64)
    }

    // Generates a random sample for the model at current dimension
    pub fn random_sample(&mut self) {
        for i in self.model.iter_mut() {
            *i = (self.rng.next_u32() % 2) as i8 * 2 - 1;
        }
    }
}

impl<const D: usize, const PBC: bool> Sample for Ising<D, PBC>
where
    [Ix; D]: IntoDimension<Dim = Dim<[Ix; D]>>,
    Dim<[Ix; D]>: Dimension,
{
    fn energy(&self) -> Energy {
        self.get_energy()
    }

    fn resample(&mut self) {
        self.random_sample();
    }
}

#[cfg(test)]
mod tests {
    use super::Ising;

    const FLOAT_CRITERIA: f64 = 1e-6;

    #[test]
    fn test_construct() {
        let ising_2d = Ising::<2, false>::new(1.0, 2.0, [2, 3]);

        assert_eq!(ising_2d.get_b(), 1.0);
        assert_eq!(ising_2d.get_j(), 2.0);
    }

    #[test]
    fn test_neighbors_pbc() {
        let ising = Ising::<3, true>::new(1.0, 1.0, [2, 1, 3]);

        assert_eq!(ising.get_neighbor(&[1, 0, 2], 0), Some([0, 0, 2]));
        assert_eq!(ising.get_neighbor(&[1, 0, 2], 1), None);
        assert_eq!(ising.get_neighbor(&[1, 0, 2], 2), Some([1, 0, 0]));

        assert_eq!(ising.get_neighbor(&[1, 0, 1], 2), Some([1, 0, 2]));
    }

    #[test]
    fn test_neighbors_no_pbc() {
        let ising = Ising::<3, false>::new(1.0, 1.0, [2, 1, 3]);

        assert_eq!(ising.get_neighbor(&[1, 0, 1], 0), None);
        assert_eq!(ising.get_neighbor(&[1, 0, 1], 1), None);
        assert_eq!(ising.get_neighbor(&[1, 0, 1], 2), Some([1, 0, 2]));
    }

    #[test]
    fn test_order_parameter() {
        let d = 99;
        let mut ising = Ising::<2, false>::new(1.0, 1.0, [d, d]);

        for i in 0..d {
            for j in 0..d {
                ising.model[(i, j)] = 1;
            }
        }
        assert!((ising.get_order_parameter() - 1.0).abs() < FLOAT_CRITERIA);

        for i in 0..d {
            for j in 0..d {
                ising.model[(i, j)] = if j % 3 == 0 { -1 } else { 1 };
            }
        }
        assert!((ising.get_order_parameter() - 1.0 / 3.0).abs() < FLOAT_CRITERIA);
    }

    #[test]
    fn test_energy_external_field() {
        let d = 99;
        let mut ising = Ising::<2, false>::new(0.5, 0.0, [d, d]);

        for i in 0..d {
            for j in 0..d {
                ising.model[(i, j)] = if j % 3 == 0 { -1 } else { 1 };
            }
        }
        // 3267 = 99 * 99 / 3
        assert!((ising.get_energy() - (-0.5) * 3267.0).abs() < FLOAT_CRITERIA);
    }

    #[test]
    fn test_energy_pairwise_no_pbc() {
        let d = 3;
        let mut ising = Ising::<2, false>::new(0.0, 0.5, [d, d]);

        for i in 0..d {
            for j in 0..d {
                ising.model[(i, j)] = if j % 3 == 0 { -1 } else { 1 };
            }
        }
        // 6.0 is the no pbc pairwise value
        assert!((ising.get_energy() - 0.5 * 6.0).abs() < FLOAT_CRITERIA);
    }

    #[test]
    fn test_energy_pairwise_pbc() {
        let d = 3;
        let mut ising = Ising::<2, true>::new(0.0, 0.5, [d, d]);

        for i in 0..d {
            for j in 0..d {
                ising.model[(i, j)] = if j % 3 == 0 { -1 } else { 1 };
            }
        }
        // 6.0 is the pbc pairwise value.
        assert!((ising.get_energy() - 0.5 * 6.0).abs() < FLOAT_CRITERIA);
    }
}
