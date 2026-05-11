use std::fmt;

use rand::RngExt;

const KILO: usize = 1000;
const MEGA: usize = 1000 * KILO;
const GIGA: usize = 1000 * MEGA;

const KIBI: usize = 1024;
const MIBI: usize = 1024 * KIBI;
const GIBI: usize = 1024 * MIBI;

pub struct NumPretty(pub usize);

impl fmt::Display for NumPretty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.0;
        let s = if n < KILO {
            format!("{}", n)
        } else if n < MEGA {
            format!("{:.1}k", n as f64 / KILO as f64)
        } else if n < GIGA {
            format!("{:.1}M", n as f64 / MEGA as f64)
        } else {
            format!("{:.1}G", n as f64 / GIGA as f64)
        };
        f.pad(s.as_str())
    }
}

pub struct SizePretty(pub usize);

impl fmt::Display for SizePretty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.0;
        let s = if n < KIBI {
            format!("{}B", n)
        } else if n < MIBI {
            format!("{:.1}KiB", n as f64 / KIBI as f64)
        } else if n < GIBI {
            format!("{:.1}MiB", n as f64 / MIBI as f64)
        } else {
            format!("{:.1}GiB", n as f64 / GIBI as f64)
        };
        f.pad(s.as_str())
    }
}

#[derive(Copy, Clone, PartialOrd, PartialEq)]
pub struct Finite(f64);

impl Finite {
    pub fn into_inner(self) -> f64 {
        self.0
    }
}

impl From<f64> for Finite {
    fn from(value: f64) -> Self {
        debug_assert!(value.is_finite());
        Self(value)
    }
}

impl From<f32> for Finite {
    fn from(value: f32) -> Self {
        Self::from(value as f64)
    }
}

impl Eq for Finite {}

impl Ord for Finite {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

pub trait IteratorExt: Iterator {
    fn argmax(self) -> Option<usize>
    where
        Self: Sized,
        Self::Item: Ord + Clone,
    {
        self.enumerate()
            .max_by_key(|(_, x)| x.clone())
            .map(|(i, _)| i)
    }

    fn cumsum(self) -> CumSum<Self>
    where
        Self: Sized,
        Self::Item: Clone + std::ops::Add<Output = Self::Item>,
    {
        CumSum {
            inner: self,
            total: None,
        }
    }

    fn sample_weighted(self, rng: &mut impl rand::Rng) -> Option<usize>
    where
        Self: Sized,
        Self::Item: Clone
            + std::ops::Add<Output = Self::Item>
            + std::ops::Mul<Output = Self::Item>
            + std::cmp::PartialOrd,
        rand::distr::StandardUniform: rand::distr::Distribution<Self::Item>,
    {
        let sum = self.cumsum().collect::<Vec<_>>();
        let n = sum.len();
        if n == 0 {
            None
        } else {
            let thresh = rng.random::<Self::Item>() * sum[n - 1].clone();
            Some(sum.iter().position(|p| *p > thresh).expect("sample failed"))
        }
    }
}

impl<I: Iterator> IteratorExt for I {}

pub struct CumSum<I: Iterator> {
    inner: I,
    total: Option<I::Item>,
}

impl<I: Iterator> Iterator for CumSum<I>
where
    I::Item: std::ops::Add<Output = I::Item> + Clone,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(total) = self.total.clone() {
            if let Some(next) = self.inner.next() {
                self.total = Some(total + next);
            } else {
                return None;
            }
        } else {
            self.total = self.inner.next();
        };
        return self.total.clone();
    }
}
