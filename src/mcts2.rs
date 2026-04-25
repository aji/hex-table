use std::{ops::ControlFlow, time::Instant};

use bumpalo::Bump;

pub trait MctsState: Copy + Sized {
    fn init() -> Self;

    /// Return Some(true) for sente win, Some(false) for gote win, and None for
    /// intermediate game states.
    fn terminal(&self) -> Option<bool>;

    /// Return true for sente win and false for gote win.
    fn rollout(&self) -> bool;

    fn children(&self) -> impl ExactSizeIterator<Item = Self>;
}

struct MctsNode<'a, S> {
    state: S,
    children: Option<&'a mut [MctsNode<'a, S>]>,
    num_wins: u32,
    num_sims: u32,
    terminal: Option<bool>,
}

#[derive(Copy, Clone, Debug)]
struct MctsBackprop {
    num_wins: u32,
    num_sims: u32,
    depth: usize,
}

impl std::ops::Add for MctsBackprop {
    type Output = MctsBackprop;

    fn add(self, rhs: Self) -> Self::Output {
        MctsBackprop {
            num_wins: self.num_wins + rhs.num_wins,
            num_sims: self.num_sims + rhs.num_sims,
            depth: self.depth.max(rhs.depth),
        }
    }
}

impl<'a, S: MctsState> MctsNode<'a, S> {
    fn new(state: S) -> MctsNode<'a, S> {
        let terminal = state.terminal();
        let rollout = state.rollout();
        let num_wins = rollout as u32;
        let num_sims = 1;
        MctsNode {
            state,
            terminal,
            num_wins,
            num_sims,
            children: None,
        }
    }

    fn uct(&self, depth: usize, ln_n: f64) -> Finite {
        let w = match depth % 2 == 0 {
            true => self.num_sims - self.num_wins,
            false => self.num_wins,
        };
        let n = self.num_sims as f64;
        ((w as f64 / n) + (1.0 * ln_n / n).sqrt()).into()
    }

    fn descend(&mut self, bump: &'a Bump, depth: usize) -> MctsBackprop {
        if let Some(sente_win) = self.terminal {
            // terminal nodes expand immediately
            return MctsBackprop {
                num_wins: 100 * (sente_win as u32),
                num_sims: 100,
                depth: depth,
            };
        }

        if let Some(ref mut children) = self.children {
            // continue selection
            let ln_n = (self.num_sims as f64).ln();
            let child = children
                .iter_mut()
                .max_by_key(|n| n.uct(depth + 1, ln_n))
                .expect("no child");
            child.iter(bump, depth + 1)
        } else {
            let it = self.state.children().map(|s| MctsNode::new(s));
            let children = bump.alloc_slice_fill_iter(it);
            let backprop = children
                .iter()
                .map(|c| MctsBackprop {
                    num_wins: c.num_wins,
                    num_sims: c.num_sims,
                    depth: depth + 1,
                })
                .reduce(|a, b| a + b)
                .expect("no child");
            self.children = Some(children);
            backprop
        }
    }

    fn iter(&mut self, bump: &'a Bump, depth: usize) -> MctsBackprop {
        let backprop = self.descend(bump, depth);
        self.num_wins += backprop.num_wins;
        self.num_sims += backprop.num_sims;
        backprop
    }

    fn best(&self) -> S {
        if let Some(_) = self.terminal {
            panic!("node is terminal");
        }

        self.children
            .as_ref()
            .expect("no children")
            .iter()
            .max_by_key(|c| c.num_sims)
            .expect("no child")
            .state
            .clone()
    }

    fn best_leaf(&self) -> S {
        if let Some(_) = self.terminal {
            return self.state;
        }

        if let Some(ref children) = self.children {
            children
                .iter()
                .max_by_key(|c| c.num_sims)
                .expect("no child")
                .best_leaf()
        } else {
            self.state
        }
    }
}

pub struct MctsStats<S> {
    pub num_wins: u32,
    pub num_sims: u32,
    pub min_depth: usize,
    pub max_depth: usize,
    pub allocated_bytes: usize,
    pub best_state: S,
    pub best_state_leaf: S,
}

pub trait MctsMonitor<S> {
    fn defer(&mut self, stats: &MctsStats<S>) -> ControlFlow<()>;
}

impl<S, F> MctsMonitor<S> for F
where
    F: FnMut(&MctsStats<S>) -> ControlFlow<()>,
{
    fn defer(&mut self, stats: &MctsStats<S>) -> ControlFlow<()> {
        (self)(stats)
    }
}

pub fn search<S: MctsState, M: MctsMonitor<S>>(state: S, depth: usize, mut monitor: M) -> S {
    let bump = Bump::new();
    let mut root = MctsNode::new(state);
    let mut last_defer: Instant = Instant::now();
    let mut min_depth: usize = std::usize::MAX;
    let mut max_depth: usize = 0;

    loop {
        for _ in 0..100 {
            let backprop = root.iter(&bump, depth);
            let stat_depth = backprop.depth - depth;
            min_depth = min_depth.min(stat_depth);
            max_depth = max_depth.max(stat_depth);
        }

        if last_defer.elapsed().as_millis() > 100 {
            let stats = MctsStats {
                num_wins: root.num_wins,
                num_sims: root.num_sims,
                min_depth,
                max_depth,
                allocated_bytes: bump.allocated_bytes(),
                best_state: root.best(),
                best_state_leaf: root.best_leaf(),
            };
            match monitor.defer(&stats) {
                ControlFlow::Continue(_) => {}
                ControlFlow::Break(_) => break,
            }
            last_defer = Instant::now();
            min_depth = std::usize::MAX;
            max_depth = 0;
        }
    }

    root.best()
}

#[derive(Copy, Clone, PartialOrd, PartialEq)]
struct Finite(f64);

impl From<f64> for Finite {
    fn from(value: f64) -> Self {
        debug_assert!(value.is_finite());
        Self(value)
    }
}

impl Eq for Finite {}

impl Ord for Finite {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
