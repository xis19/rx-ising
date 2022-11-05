use std::collections::HashMap;

use super::{
    ensemble_average::EnsembleAverage,
    metropolis::Metropolis,
    sample::Sample,
    sample_counter::SampleCounter,
    types::{Energy, Temperature},
};

pub struct MonteCarlo<SampleType>
where
    SampleType: Sample,
{
    sample_counter: SampleCounter,
    ensemble_averages: HashMap<String, EnsembleAverage>,
    metropolis: Metropolis,
    temperature: Temperature,
    sample: SampleType,
    last_sample_energy: Energy,
}

impl<SampleType> MonteCarlo<SampleType>
where
    SampleType: Sample,
{
    pub fn new(k_b: f64, temperature: Temperature, sample: SampleType) -> Self {
        let mut result = MonteCarlo {
            sample_counter: SampleCounter::new(),
            ensemble_averages: HashMap::<String, EnsembleAverage>::new(),
            metropolis: Metropolis::new(k_b),
            temperature,
            sample,
            // Ensures the first sample is accepted
            last_sample_energy: Energy::MAX,
        };

        result
            .ensemble_averages
            .insert("Energy".to_string(), EnsembleAverage::new());

        result
    }

    pub fn get_sample(&self) -> &SampleType {
        &self.sample
    }

    pub fn get_sample_counter(&self) -> &SampleCounter {
        &self.sample_counter
    }

    pub fn get_ensemble_average(&self, name: &str) -> Option<&EnsembleAverage> {
        self.ensemble_averages.get(name)
    }

    pub fn set_ensemble_average(&mut self, name: &str, value: f64) {
        if !self.ensemble_averages.contains_key(name) {
            self.ensemble_averages
                .insert(name.to_string(), EnsembleAverage::new());
        }
        self.ensemble_averages
            .get_mut(name)
            .unwrap()
            .accept_sample(value);
    }

    // Generates new samples until one being accepted, updates the corresponding statistics information
    pub fn next_sample(&mut self) {
        loop {
            self.sample.resample();
            let sample_energy = self.sample.energy();
            if self
                .metropolis
                .apply(&sample_energy, &self.last_sample_energy, &self.temperature)
            {
                self.sample_counter.accept();
                self.ensemble_averages
                    .get_mut("Energy")
                    .expect("Must have Energy term")
                    .accept_sample(sample_energy);
                self.last_sample_energy = sample_energy;
                break;
            } else {
                self.sample_counter.reject();
            }
        }
    }
}
