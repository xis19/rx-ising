use super::types::{Energy, Temperature};

use rand::prelude::*;

pub struct Metropolis {
    k_b: f64,
    rng: ThreadRng,
}

impl Metropolis {
    pub fn new(k_b: f64) -> Self {
        Metropolis {
            k_b,
            rng: thread_rng(),
        }
    }

    /// Applies a metropolis acceptance/rejection algorithm
    pub fn apply(
        &mut self,
        new_sample_energy: &Energy,
        old_sample_energy: &Energy,
        temperature: &Temperature,
    ) -> bool {
        // If the energy of the sample is lower, the new sample is always accepted.
        if new_sample_energy < old_sample_energy {
            return true;
        }

        // Otherwise, conditionally accepts the new sample with a probability.
        let kbt_inverse = 1.0 / (self.k_b * temperature);
        let e_new = -new_sample_energy * kbt_inverse;
        let e_old = -old_sample_energy * kbt_inverse;
        let e_diff = e_new - e_old;
        let rand01 = (self.rng.gen_range(0..i32::MAX) as f64) / (i32::MAX as f64);

        e_diff.exp() > rand01
    }
}

#[cfg(test)]
mod tests {
    use super::Metropolis;

    /// Tests if the new energy is lower than old energy, Metropolis will always accept the configuration
    /// FIXME This test is incomplete since only a few samples are used.
    #[test]
    fn test_acceptance() {
        let mut metropolis = Metropolis::new(1.0);

        assert!(metropolis.apply(&1.0, &2.0, &298.0));
        assert!(metropolis.apply(&1.0, &4.0, &298.0));
    }

    /// Tests if the new energy is higher than the old energy, Metropolis will conditionally accept the configuration
    #[test]
    fn test_conditionally_acceptance() {
        let temperature = 1.0;
        let old_energy = 10.0;
        let new_energy = 10.01;
        let acceptance_rate = (-(new_energy - old_energy) as f64).exp();
        let mut metropolis = Metropolis::new(1.0);
        let total_tests = 1000000;
        let mut acceptance = 0;
        let float_error_criteria = 0.001;

        for _ in 0..total_tests {
            acceptance += match metropolis.apply(&new_energy, &old_energy, &temperature) {
                true => 1,
                false => 0,
            }
        }

        assert!(
            ((acceptance as f64) / (total_tests as f64) - acceptance_rate).abs()
                < float_error_criteria
        );
    }
}
