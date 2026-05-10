use std::{
    ops::ControlFlow,
    sync::{Arc, Mutex, MutexGuard},
    time::Instant,
};

use crate::{
    bb::Bitboard,
    mcts::{self, MctsStats},
    util::{NumPretty, SizePretty},
};

pub struct AgentState {
    pub status: AgentStatus,
    pub best_state: Option<Bitboard>,
}

pub enum AgentStatus {
    Idle,
    Thinking,
}

pub enum AgentMessage {
    BoardChanged(Bitboard, usize),
}

pub trait AgentThinker: Clone + Send + 'static {
    fn think(self, handle: ThinkHandle);
}

impl AgentThinker for fn(ThinkHandle) {
    fn think(self, handle: ThinkHandle) {
        (self)(handle)
    }
}

pub struct Agent<T> {
    thinker: T,
}

impl<T: AgentThinker> Agent<T> {
    pub fn new(thinker: T) -> Self {
        Self { thinker }
    }

    pub fn think(&self, board: Bitboard, turn: usize) -> ThinkHandle {
        let handle = ThinkHandle::new(board, turn);
        std::thread::spawn({
            let thinker = self.thinker.clone();
            let handle = handle.clone();
            move || thinker.think(handle)
        });
        handle
    }
}

pub fn mcts_thinking_task(task: ThinkHandle) {
    let (board, turn) = {
        let data = task.data();
        (data.board, data.turn)
    };

    let out = mcts::search(board, turn, {
        let start = Instant::now();
        let task = task.clone();

        move |stats: &MctsStats<Bitboard>| {
            let elapsed = start.elapsed();
            let mut task = task.data();

            task.message = {
                let msg = format!(
                    "{:>10} {:5.3} d={:2}..{:2} {:>9}({:>4}/sim) {:>7.1?}({:4?}/sim)",
                    NumPretty(stats.num_sims as usize),
                    stats.num_wins as f32 / stats.num_sims as f32,
                    stats.min_depth,
                    stats.max_depth,
                    SizePretty(stats.allocated_bytes),
                    SizePretty(stats.allocated_bytes / stats.num_sims as usize),
                    elapsed,
                    elapsed / stats.num_sims
                );
                Some(msg)
            };

            let stop_on_iters = stats.num_sims >= 500_000_000;
            let stop_on_time = elapsed.as_secs() >= 2;
            let stop_on_aborted = task.aborted;
            match stop_on_iters || stop_on_time || stop_on_aborted {
                true => ControlFlow::Break(()),
                false => ControlFlow::Continue(()),
            }
        }
    });

    let _ = {
        let mut data = task.data();
        data.result = Some(out.best);
    };
}

#[derive(Clone)]
pub struct ThinkHandle {
    data: Arc<Mutex<ThinkData>>,
}

pub struct ThinkData {
    pub board: Bitboard,
    pub turn: usize,
    pub message: Option<String>,
    pub result: Option<Bitboard>,
    pub aborted: bool,
}

impl ThinkHandle {
    pub fn new(board: Bitboard, turn: usize) -> ThinkHandle {
        let data = ThinkData {
            board,
            turn,
            message: None,
            result: None,
            aborted: false,
        };
        ThinkHandle {
            data: Arc::new(Mutex::new(data)),
        }
    }

    pub fn data(&'_ self) -> MutexGuard<'_, ThinkData> {
        self.data.lock().unwrap()
    }

    pub fn message(&self) -> Option<String> {
        self.data().message.clone()
    }

    pub fn result(&self) -> Option<Bitboard> {
        self.data().result.clone()
    }

    pub fn abort(&self) {
        self.data().aborted = true;
    }
}
