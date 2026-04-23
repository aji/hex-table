use std::{
    cmp::Ordering,
    f32,
    io::Write,
    time::{Duration, Instant},
};

const PRINT_INTERVAL: Duration = Duration::from_millis(200);
const EXPLORE: f32 = f32::consts::SQRT_2;

pub trait MctsState: Sized {
    fn init() -> Self;

    fn terminal(&self) -> Option<bool>;

    fn rollout(&self) -> bool;

    fn children(&self) -> impl Iterator<Item = Self>;
}

pub struct MctsTree<S> {
    last_print: Instant,
    root_depth: usize,
    root: MctsNode<S>,
}

enum MctsChildren<S> {
    Unknown,
    Leaf(bool),
    List(Vec<MctsNode<S>>),
}

struct MctsNode<S> {
    state: S,
    sente_wins: usize,
    rollouts: usize,
    children: MctsChildren<S>,
}

impl<S: MctsState> MctsTree<S> {
    pub fn new() -> MctsTree<S> {
        MctsTree {
            last_print: Instant::now(),
            root_depth: 0,
            root: MctsNode {
                state: S::init(),
                sente_wins: 0,
                rollouts: 1,
                children: MctsChildren::Unknown,
            },
        }
    }

    pub fn state(&self) -> &S {
        &self.root.state
    }

    pub fn size(&self) -> usize {
        self.root.rollouts
    }

    pub fn iter(&mut self) {
        self.root.iter(self.root_depth);
        let now = Instant::now();
        if now - self.last_print > PRINT_INTERVAL {
            self.last_print = now;
            print!(
                "\x1b[G{:5}{:10}/{:10}  {:10.6}",
                self.root_depth,
                self.root.sente_wins,
                self.root.rollouts,
                self.root.sente_wins as f32 / self.root.rollouts as f32
            );
            std::io::stdout().flush().unwrap();
        }
    }

    pub fn into_best(mut self) -> Result<Self, Self> {
        match self.root.into_best(self.root_depth) {
            Ok(root) => {
                self.root = root;
                self.root_depth += 1;
                Ok(self)
            }
            Err(root) => {
                self.root = root;
                Err(self)
            }
        }
    }
}

impl<S: MctsState> MctsNode<S> {
    fn new(state: S) -> MctsNode<S> {
        if let Some(sente_win) = state.terminal() {
            MctsNode {
                state,
                sente_wins: sente_win as usize,
                rollouts: 1,
                children: MctsChildren::Leaf(sente_win),
            }
        } else {
            let sente_win = state.rollout();
            MctsNode {
                state,
                sente_wins: sente_win as usize,
                rollouts: 1,
                children: MctsChildren::Unknown,
            }
        }
    }

    fn uct(&self, depth: usize, ln_n: f32, explore: f32) -> F32Ord {
        let sente = depth % 2 == 0;
        let wins = match sente {
            true => self.sente_wins,
            false => self.rollouts - self.sente_wins,
        };
        let n = self.rollouts as f32;
        (wins as f32 / n + explore * (ln_n / n).sqrt()).into()
    }

    fn iter(&mut self, depth: usize) {
        match self.children {
            MctsChildren::Unknown => {
                self.children =
                    MctsChildren::List(self.state.children().map(MctsNode::new).collect());
            }
            MctsChildren::Leaf(sente_win) => {
                self.sente_wins += sente_win as usize;
                self.rollouts += 1;
                return;
            }
            MctsChildren::List(ref mut children) => {
                let ln_n = (self.rollouts as f32).ln();
                let child = children
                    .iter_mut()
                    .max_by_key(|c| c.uct(depth, ln_n, EXPLORE))
                    .expect("no children");
                child.iter(depth + 1);
            }
        }
        let MctsChildren::List(ref children) = self.children else {
            panic!("no children");
        };
        self.sente_wins = 0;
        self.rollouts = 0;
        for child in children.iter() {
            self.sente_wins += child.sente_wins;
            self.rollouts += child.rollouts;
        }
    }

    fn into_best(self, depth: usize) -> Result<Self, Self> {
        let ln_n = (self.rollouts as f32).ln();
        let children = match self.children {
            MctsChildren::Unknown => panic!(),
            MctsChildren::Leaf(_) => return Err(self),
            MctsChildren::List(children) => children,
        };
        let child = children
            .into_iter()
            .max_by_key(|c| c.uct(depth, ln_n, 0.0))
            .expect("no children");
        Ok(child)
    }
}

#[derive(Copy, Clone, PartialOrd)]
struct F32Ord(f32);

impl From<f32> for F32Ord {
    fn from(value: f32) -> Self {
        assert!(value.is_finite());
        Self(value)
    }
}

impl PartialEq for F32Ord {
    fn eq(&self, other: &Self) -> bool {
        self.0.partial_cmp(&other.0).unwrap() == Ordering::Equal
    }
}

impl Eq for F32Ord {}

impl Ord for F32Ord {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}
