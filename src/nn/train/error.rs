use std::{fmt, ops::ControlFlow};

#[derive(Debug)]
pub enum TrainError {
    Http(reqwest::Error),
    Other(String),
}

pub type TrainResult<T> = Result<T, TrainError>;

impl TrainError {
    pub fn is_retryable(&self) -> bool {
        if let TrainError::Http(e) = self {
            match e.status() {
                Some(s) if !s.is_server_error() => false,
                _ => true,
            }
        } else {
            false
        }
    }

    pub fn continue_if_retryable(this: TrainError) -> ControlFlow<TrainError, ()> {
        log::error!("{this}");
        match this.is_retryable() {
            true => ControlFlow::Continue(()),
            false => ControlFlow::Break(this),
        }
    }

    pub fn unrecoverable<T>(this: TrainError) -> T {
        log::error!("unrecoverable: {this}");
        std::process::exit(15)
    }
}

impl fmt::Display for TrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrainError::Http(e) => write!(f, "http: {e}"),
            TrainError::Other(e) => write!(f, "{e}"),
        }
    }
}

impl From<reqwest::Error> for TrainError {
    fn from(value: reqwest::Error) -> Self {
        Self::Http(value)
    }
}

impl From<String> for TrainError {
    fn from(value: String) -> Self {
        Self::Other(value)
    }
}

impl<'a> From<&'a str> for TrainError {
    fn from(value: &'a str) -> Self {
        Self::Other(value.to_owned())
    }
}
