use std::fmt;

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
