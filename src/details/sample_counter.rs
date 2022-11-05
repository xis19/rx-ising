pub struct SampleCounter {
    generated: usize,
    accepted: usize,
}

impl SampleCounter {
    pub fn new() -> Self {
        SampleCounter {
            generated: 0,
            accepted: 0,
        }
    }

    pub fn accept(&mut self) {
        self.generated += 1;
        self.accepted += 1;
    }

    pub fn reject(&mut self) {
        self.generated += 1;
    }

    pub fn get_generated(&self) -> usize {
        self.generated
    }

    pub fn get_accepted(&self) -> usize {
        self.accepted
    }

    pub fn get_acceptance_rate(&self) -> f64 {
        self.accepted as f64 / self.generated as f64
    }
}

#[cfg(test)]
mod tests {
    use super::SampleCounter;

    const FLOAT_CRITERIA: f64 = 1e-6;

    #[test]
    fn test_sample_counter() {
        let mut counter = SampleCounter::new();
        for i in 0..99999 {
            if i % 3 == 0 {
                counter.reject();
            } else {
                counter.accept();
            }
        }
        assert_eq!(counter.get_accepted(), 66666);
        assert_eq!(counter.get_generated(), 99999);
        assert!((counter.get_acceptance_rate() - 2.0 / 3.0).abs() < FLOAT_CRITERIA);
    }
}
