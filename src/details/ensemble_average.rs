pub struct EnsembleAverage {
    count: usize,
    average: f64,
    variance: f64,
    standard_derivation: f64,

    sum: f64,
    square_sum: f64,
}

impl EnsembleAverage {
    pub fn new() -> Self {
        EnsembleAverage {
            count: 0,
            average: 0.0,
            variance: 0.0,
            standard_derivation: 0.0,
            sum: 0.0,
            square_sum: 0.0,
        }
    }

    pub fn get_average(&self) -> f64 {
        self.average
    }

    pub fn get_standard_derivation(&self) -> f64 {
        self.standard_derivation
    }

    pub fn get_variance(&self) -> f64 {
        self.variance
    }

    pub fn accept_sample(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.square_sum += value * value;
        self.average = self.sum / (self.count as f64);
        self.variance = -self.average.powi(2) + 1.0 / self.count as f64 * self.square_sum;
        self.standard_derivation = self.variance.sqrt();
    }
}

#[cfg(test)]
mod tests {
    use super::EnsembleAverage;

    const FLOAT_CRITERIA: f64 = 1e-6;

    #[test]
    fn test_ensemble() {
        let mut ensemble_average = EnsembleAverage::new();

        ensemble_average.accept_sample(1.0);
        assert!((ensemble_average.get_average() - 1.0).abs() < FLOAT_CRITERIA);
        assert!((ensemble_average.get_variance() - 0.0).abs() < FLOAT_CRITERIA);
        assert!((ensemble_average.get_standard_derivation() - 0.0).abs() < FLOAT_CRITERIA);

        ensemble_average.accept_sample(2.0);
        assert!((ensemble_average.get_average() - 1.5).abs() < FLOAT_CRITERIA);
        assert!((ensemble_average.get_variance() - 0.25).abs() < FLOAT_CRITERIA);
        assert!((ensemble_average.get_standard_derivation() - 0.5).abs() < FLOAT_CRITERIA);

        ensemble_average.accept_sample(5.0);
        assert!((ensemble_average.get_average() - 2.666666666).abs() < FLOAT_CRITERIA);
        assert!((ensemble_average.get_variance() - 2.888888889).abs() < FLOAT_CRITERIA);
        assert!((ensemble_average.get_standard_derivation() - 1.699673).abs() < FLOAT_CRITERIA);
    }
}
