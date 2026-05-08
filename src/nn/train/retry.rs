use std::{ops::ControlFlow, time::Duration};

pub const DEFAULT_RETRY: RetryStrategy = RetryStrategy {};

#[derive(Copy, Clone)]
pub struct RetryStrategy {
    // no fields
}

impl RetryStrategy {
    pub fn attempt<T, E, F, G>(&self, body: F, inspect_error: G) -> Result<T, E>
    where
        F: Fn() -> Result<T, E>,
        G: Fn(E) -> ControlFlow<E, ()>,
    {
        for delay in self.into_iter() {
            if let Some(delay) = delay {
                log::warn!("retrying after {delay:?}");
                std::thread::sleep(delay);
            }
            match body() {
                Ok(res) => return Ok(res),
                Err(e) => match inspect_error(e) {
                    ControlFlow::Break(e) => return Err(e),
                    ControlFlow::Continue(_) => continue,
                },
            }
        }
        panic!()
    }
}

impl IntoIterator for RetryStrategy {
    type Item = Option<Duration>;

    type IntoIter = Retry;

    fn into_iter(self) -> Self::IntoIter {
        Retry::new()
    }
}

pub struct Retry {
    first: bool,
}

impl Retry {
    pub fn new() -> Retry {
        Retry { first: true }
    }
}

impl Iterator for Retry {
    type Item = Option<Duration>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.first {
            self.first = false;
            Some(None)
        } else {
            Some(Some(Duration::from_millis(1000)))
        }
    }
}
