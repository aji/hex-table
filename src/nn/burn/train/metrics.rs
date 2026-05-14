use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

pub struct CounterLog {
    log: Vec<(Instant, usize)>,
}

impl CounterLog {
    pub fn new() -> CounterLog {
        CounterLog {
            log: vec![(Instant::now(), 0)],
        }
    }

    pub fn report(&mut self, value: &AtomicUsize) {
        let now = Instant::now();
        self.log.push((now, value.load(Ordering::Relaxed)));
        while (now - self.log[0].0).as_secs() > 600 {
            self.log.remove(0);
        }
    }

    pub fn latest(&self) -> usize {
        self.log.last().unwrap().1
    }

    pub fn per_second(&self) -> f64 {
        let n = self.log.len();
        if n < 2 {
            return f64::NAN;
        }
        let (t0, x0) = self.log[0];
        let (tn, xn) = self.log[n - 1];
        (xn - x0) as f64 / (tn - t0).as_secs_f64()
    }

    pub fn seconds_per(&self) -> f64 {
        1.0 / self.per_second()
    }

    pub fn count(&self) -> usize {
        self.log[self.log.len() - 1].1 - self.log[0].1
    }
}
