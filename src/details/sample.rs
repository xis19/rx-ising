use super::types::Energy;

pub trait Sample {
    // Generate a new sample
    fn resample(&mut self);

    // Returns the energy for the Sample
    fn energy(&self) -> Energy;
}
