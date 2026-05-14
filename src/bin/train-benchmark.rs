use std::{
    f64::consts::PI,
    fs::File,
    io::{self, Write},
    ops::ControlFlow,
    path::{Path, PathBuf},
    sync::LazyLock,
    time::{Duration, Instant},
};

use burn::tensor::backend::Backend;
use clap::{Parser, Subcommand};
use hex_table::{
    bb::Bitboard,
    mcts::{self, MctsMonitor, MctsStats},
    nn::{
        burn::model::BurnModel,
        model::{EvalRequest, Model, ModelConfig},
    },
    util::{Finite, IteratorExt},
};
use tqdm::Iter;

type Wgpu = burn::backend::Wgpu<f32, i32>;

#[derive(Parser, Debug)]
struct Cli {
    /// The model directory
    #[arg(long, value_name = "DIR")]
    model_dir: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Compare(CompareCommand),
    Rank(RankCommand),
}

fn main() -> io::Result<()> {
    env_logger::init();

    let cli = Cli::parse();
    log::info!("got options: {cli:?}");

    let device = Default::default();
    let config =
        ModelConfig::load(cli.model_dir.join("config.json")).expect("could not load model config");
    log::info!("loaded model config: {config:?}");
    let model: BurnModel<Wgpu> = config.init(&device);

    let checkpoints = {
        let mut checkpoints = std::fs::read_dir(&cli.model_dir)
            .expect("could not read model dir")
            .map(|x| x.expect("could not read dir item"))
            .flat_map(|x| x.file_name().into_string().into_iter())
            .filter(|x| x.starts_with("checkpoint-"))
            .collect::<Vec<_>>();
        checkpoints.sort();
        checkpoints
    };
    log::info!("found {} checkpoints", checkpoints.len());

    match cli.command {
        Commands::Compare(ref cmd) => cmd_compare(&cli, cmd, checkpoints, model, device),
        Commands::Rank(ref cmd) => cmd_rank(&cli, cmd, checkpoints, model, device),
    }
}

/// Compare model checkpoints to an MCTS benchmark player
#[derive(Parser, Debug)]
struct CompareCommand {
    /// The CSV file to write results to
    #[arg(long, value_name = "FILE", default_value = "compare-mcts.csv")]
    output: PathBuf,

    /// The number of games to play per eval
    #[arg(long, value_name = "N", default_value = "50")]
    games: usize,

    /// The number of recent checkpoints to use. If omitted, all checkpoints are used.
    #[arg(long, value_name = "N")]
    checkpoints: Option<usize>,

    /// Multiplies the MCTS player's CPU allocation by 10^F
    #[arg(long, value_name = "F", default_value = "0.0")]
    mcts_handicap: f64,
}

fn cmd_compare<B: Backend>(
    cli: &Cli,
    cmd: &CompareCommand,
    mut checkpoints: Vec<String>,
    mut model: BurnModel<B>,
    device: B::Device,
) -> io::Result<()> {
    if let Some(n) = cmd.checkpoints
        && n < checkpoints.len()
    {
        log::info!("using only {n} most recent checkpoints");
        let _ = checkpoints.drain(..checkpoints.len() - n);
    }

    let nn_evals_per_time = bench_model_evals(&model, &device, false);
    let mcts_evals_per_time = bench_mcts_evals(false);

    let mcts_per_nn_base = (mcts_evals_per_time / nn_evals_per_time).round() as u32;
    let mcts_per_nn =
        (10.0f64.powf(cmd.mcts_handicap) * mcts_evals_per_time / nn_evals_per_time).round() as u32;
    log::info!(
        "roughly {} mcts evals per nn eval ({} with handicap)",
        mcts_per_nn_base,
        mcts_per_nn
    );

    let mut out = std::fs::OpenOptions::new()
        .truncate(true)
        .create(true)
        .write(true)
        .open(&cmd.output)?;
    writeln!(
        out,
        "{},{},{},{},{},{},{}",
        "checkpoint",
        "games",
        "mcts_per_nn",
        "mcts_handicap",
        "win_rate",
        "wins_as_sente",
        "wins_as_gote",
    )?;
    out.flush().ok();

    log::info!("running {} games per checkpoint", cmd.games);
    if !cmd.games.is_multiple_of(2) {
        log::warn!(
            "odd number of games per checkpoint is not recommended due to slight first-player advantage"
        );
    }
    for checkpoint in checkpoints.iter().tqdm() {
        let bytes = std::fs::read(cli.model_dir.join(checkpoint))?;
        model = model.load_bytes(bytes, &device);
        let (win_rate, wins_as_sente, wins_as_gote) =
            make_them_fight(&model, &device, cmd.games, mcts_per_nn);
        writeln!(
            out,
            "{},{},{},{},{},{},{}",
            checkpoint,
            cmd.games,
            mcts_per_nn,
            cmd.mcts_handicap,
            win_rate,
            wins_as_sente,
            wins_as_gote,
        )?;
        out.flush().ok();
    }

    Ok(())
}

/// Rank the latest model checkpoint relative to MCTS
#[derive(Parser, Debug)]
struct RankCommand {
    /// Use a specific checkpoint
    #[arg(long, value_name = "FILE")]
    checkpoint: Option<PathBuf>,

    /// Rank all checkpoints and write their ranks to the given CSV
    #[arg(long, value_name = "FILE")]
    rank_all: Option<PathBuf>,

    /// Stop at the given stddev cutoff
    #[arg(long, value_name = "X")]
    stddev_stop: Option<f64>,

    /// Stop at the given number of iterations
    #[arg(long, value_name = "N")]
    iters_stop: Option<usize>,

    /// Minimum rank. Defaults to 1
    #[arg(long, value_name = "N", default_value = "1.0")]
    rank_min: f64,

    /// Maximum rank. Defaults to 8
    #[arg(long, value_name = "N", default_value = "8.0")]
    rank_max: f64,
}

fn cmd_rank<B: Backend>(
    cli: &Cli,
    cmd: &RankCommand,
    checkpoints: Vec<String>,
    model: BurnModel<B>,
    device: B::Device,
) -> io::Result<()> {
    let nn_evals_per_time = bench_model_evals(&model, &device, true);
    let mcts_evals_per_time = bench_mcts_evals(true);
    let compute_equiv_rank =
        ((mcts_evals_per_time / nn_evals_per_time).log10() * 100.0).round() / 100.0;
    assert!(compute_equiv_rank > 0.0);

    let mut rank_one = RankOne {
        model: &model,
        device: &device,
        compute_equiv_rank,
        stddev_stop: cmd.stddev_stop,
        iters_stop: cmd.iters_stop,
        rank_min: Some(cmd.rank_min),
        rank_max: Some(cmd.rank_max),
    };

    if rank_one.iters_stop.is_none() && rank_one.stddev_stop.is_none() {
        log::warn!("using default stopping condition of 40 iters");
        rank_one.iters_stop = Some(40);
    }

    if let Some(ref path) = cmd.rank_all {
        let mut out = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)?;
        writeln!(
            out,
            "{},{},{},{},{},{},{}",
            "checkpoint",
            "rank",
            "compute_equiv_rank",
            "mean",
            "stddev",
            "iters",
            "elapsed_seconds"
        )?;
        out.flush().ok();
        for checkpoint in checkpoints {
            let path = cli.model_dir.join(checkpoint);
            cmd_rank_one(Some(&mut out), &path, &rank_one)?;
        }
    } else if let Some(ref path) = cmd.checkpoint {
        cmd_rank_one(None, path, &rank_one)?;
    } else {
        let checkpoint = checkpoints
            .into_iter()
            .last()
            .ok_or_else(|| io::Error::other("no checkpoints"))?;
        let path = cli.model_dir.join(checkpoint);
        cmd_rank_one(None, &path, &rank_one)?;
    };

    Ok(())
}

struct RankOne<'a, B: Backend> {
    model: &'a BurnModel<B>,
    device: &'a B::Device,
    compute_equiv_rank: f64,
    stddev_stop: Option<f64>,
    iters_stop: Option<usize>,
    rank_min: Option<f64>,
    rank_max: Option<f64>,
}

fn cmd_rank_one<'a, B: Backend>(
    out: Option<&mut File>,
    checkpoint: &Path,
    cmd: &'a RankOne<'a, B>,
) -> io::Result<()> {
    const RANK_SUBUNITS: f64 = 128.0;

    let rank_min = cmd.rank_min.unwrap_or(1.0);
    let rank_max = cmd.rank_max.unwrap_or(8.0);

    let ranks_n = ((rank_max - rank_min) * RANK_SUBUNITS + 1.0).round() as usize;
    let ranks_xs = Linspace::new(rank_min as f64, rank_max as f64, ranks_n);
    let mut ranks = Prior::from_fn(ranks_xs, |x| {
        let uniform = 1.0 / ranks_n as f64;
        let normal = x.normal(cmd.compute_equiv_rank, 1.0);
        0.75.lerp(normal, uniform)
    });

    let start = Instant::now();
    let model = {
        log::info!("loading {}", checkpoint.display());
        let bytes = std::fs::read(checkpoint)?;
        cmd.model.clone().load_bytes(bytes, cmd.device)
    };

    let mut model_player = ModelPlayer::new(&model, &cmd.device);
    for iter in 0usize.. {
        let rank = ranks.argmax();
        let stats = ranks.stats();

        let handicap = rank - cmd.compute_equiv_rank;
        let stddev = stats.variance.sqrt();
        let strength = 10.0f64.powf(rank) as u32;

        ranks.show(1.0);
        log::info!(
            "mean={:.2} stddev={:.2} rank={:.2}, handicap={:.2} strength={strength}",
            stats.mean,
            stddev,
            rank,
            handicap,
        );

        let iters_stop = cmd.iters_stop.map(|x| x <= iter).unwrap_or(false);
        let stddev_stop = cmd.stddev_stop.map(|x| stddev <= x).unwrap_or(false);
        if iters_stop || stddev_stop {
            let rank = ranks.argmax();
            let handicap = rank - cmd.compute_equiv_rank;
            let lo = 10.0f64.powf(handicap - stddev * 2.0);
            let hi = 10.0f64.powf(handicap + stddev * 2.0);
            log::info!("model seems {lo:.1}x-{hi:.1}x as fast as mcts",);
            if let Some(out) = out {
                writeln!(
                    out,
                    "{},{:.2},{:.2},{:.5},{:.5},{},{}",
                    checkpoint.file_name().unwrap().display(),
                    rank,
                    cmd.compute_equiv_rank,
                    stats.mean,
                    stddev,
                    iter,
                    start.elapsed().as_secs_f64()
                )?;
                out.flush().ok();
            }
            break;
        }

        let mut mcts_player = MctsPlayer::new(strength);
        let model_is_sente = iter.is_multiple_of(2);
        let sente_win = match model_is_sente {
            true => play(&mut model_player, &mut mcts_player),
            false => play(&mut mcts_player, &mut model_player),
        };
        ranks.update(|x| {
            let sente_relative_rank = match model_is_sente {
                true => x - rank,
                false => rank - x,
            };
            let p = p_sente_win(sente_relative_rank);
            match sente_win {
                true => p,
                false => 1.0 - p,
            }
        });
    }

    Ok(())
}

fn p_sente_win(sente_relative_rank: f64) -> f64 {
    // This is roughly calibrated to the logistic regression in skill.ipynb
    (0.3 + 1.8 * sente_relative_rank).sigmoid()
}

#[derive(Copy, Clone, Debug)]
struct Linspace {
    start: f64,
    stop: f64,
    count: usize,
}

impl Linspace {
    fn new(start: f64, stop: f64, count: usize) -> Linspace {
        Linspace { start, stop, count }
    }

    fn nth(&self, i: usize) -> f64 {
        (i as f64 / (self.count - 1) as f64).lerp(self.start, self.stop)
    }
}

/// A discrete prior distribution
#[derive(Clone)]
struct Prior {
    xs: Linspace,
    ys: Vec<f64>,
}

struct PriorStats {
    pub mean: f64,
    pub variance: f64,
}

const SCALE: LazyLock<Vec<char>> = LazyLock::new(|| SCALE_CHARS.chars().collect());
const SCALE_CHARS: &'static str =
    " \u{2581}\u{2582}\u{2583}\u{2584}\u{2585}\u{2586}\u{2587}\u{2588}";

impl Prior {
    fn from_data(xs: Linspace, ys: Vec<f64>) -> Prior {
        assert_eq!(ys.len(), xs.count);
        (Prior { xs, ys }).into_normalized()
    }

    fn from_fn<F>(xs: Linspace, f: F) -> Prior
    where
        F: Fn(f64) -> f64,
    {
        Self::from_data(xs, (0..xs.count).map(|i| f(xs.nth(i))).collect())
    }

    fn iter(&self) -> impl Iterator<Item = Finite> {
        self.ys.iter().copied().map(Finite::from)
    }

    fn show(&self, ticks: f64) {
        let max = self.iter().max().unwrap().into_inner();
        let chunk_size = self.ys.len() / 100;
        let scaled = self
            .ys
            .chunks_exact(chunk_size)
            .map(|x| x.iter().copied().sum::<f64>() / (chunk_size as f64 * max))
            .collect::<Vec<_>>();
        let rows = 2;
        for r in 0..rows {
            for y in scaled.iter().copied() {
                let y0 = (rows - r - 1) as f64 / rows as f64;
                let y1 = (rows - r) as f64 / rows as f64;
                let s = y.unlerp(y0, y1).lerp(0.0, (SCALE.len() - 1) as f64).round();
                let s = (s as usize).clamp(0, SCALE.len() - 1);
                print!("{}", SCALE[s]);
            }
            println!();
        }
        let mut last_tick = self.xs.nth(0) - ticks;
        for i in 0..scaled.len() + 1 {
            let i = i * chunk_size;
            let x = self.xs.nth(i);
            if x - last_tick >= ticks {
                print!("'");
                last_tick += ticks;
            } else {
                print!(" ");
            }
        }
        println!()
    }

    fn stats(&self) -> PriorStats {
        let mut mean_num = 0.0;
        let mut mean_den = 0.0;

        for (i, y) in self.ys.iter().enumerate() {
            let x = self.xs.nth(i);
            mean_num += y * x;
            mean_den += y;
        }

        // total_den should be 1.0 but just in case...
        let mean = mean_num / mean_den;

        let mut var_num = 0.0;
        let mut var_den = 0.0;
        for (i, y) in self.ys.iter().enumerate() {
            let x = self.xs.nth(i);
            let z = (x - mean).powi(2);
            var_num += y * z;
            var_den += y;
        }

        // var_den should be 1.0 but just in case...
        let variance = var_num / var_den;

        PriorStats { mean, variance }
    }

    fn update<F>(&mut self, likelihood: F)
    where
        F: Fn(f64) -> f64,
    {
        self.ys
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x *= likelihood(self.xs.nth(i)));
        self.normalize();
    }

    fn argmax(&self) -> f64 {
        let i = self.iter().argmax().unwrap();
        self.xs.nth(i)
    }

    fn sample(&self) -> f64 {
        let i = self
            .ys
            .iter()
            .copied()
            .sample_weighted(&mut rand::rng())
            .unwrap();
        self.xs.nth(i)
    }

    fn normalize(&mut self) {
        let sum: f64 = self.ys.iter().sum::<f64>();
        self.ys.iter_mut().for_each(|x| *x /= sum);
    }

    fn into_normalized(mut self) -> Prior {
        self.normalize();
        self
    }
}

struct MctsSims(u32);

impl<S> MctsMonitor<S> for MctsSims {
    fn defer(&mut self, stats: &MctsStats<S>) -> ControlFlow<()> {
        use ControlFlow::*;
        match stats.num_sims < self.0 {
            true => Continue(()),
            false => Break(()),
        }
    }
}

fn bench_model_evals<B: Backend>(model: &BurnModel<B>, device: &B::Device, fast: bool) -> f64 {
    let board = Bitboard::new();

    log::info!("benchmarking model evals");

    let dur = match fast {
        true => Duration::from_secs(1),
        false => Duration::from_secs(30),
    };
    log::info!("warming up for {dur:?}");
    let start = Instant::now();
    while start.elapsed() < dur {
        std::hint::black_box(model.eval_one(EvalRequest::new(board), device));
    }

    let dur = match fast {
        true => Duration::from_secs(1),
        false => Duration::from_secs(5),
    };
    log::info!("counting evals in {dur:?}");
    let mut count = 0;
    let start = Instant::now();
    while start.elapsed() < dur {
        std::hint::black_box(model.eval_one(EvalRequest::new(board), device));
        count += 1;
    }

    let res = count as f64 / dur.as_secs_f64();
    log::info!("result: {} evals per time", res);
    res
}

fn bench_mcts_evals(fast: bool) -> f64 {
    let board = Bitboard::new();

    log::info!("benchmarking mcts evals");

    let dur = match fast {
        true => Duration::from_secs(1),
        false => Duration::from_secs(5),
    };
    log::info!("counting evals in {dur:?}");
    let mut count = 0;
    let start = Instant::now();
    while start.elapsed() < dur {
        std::hint::black_box(mcts::search(board, 0, MctsSims(10000)));
        count += 10000;
    }

    let res = count as f64 / dur.as_secs_f64();
    log::info!("result: {} evals per time", res);
    res
}

trait Player {
    fn record_win(&mut self, sente: bool);
    fn play(&self, board: Bitboard) -> Bitboard;
}

struct ModelPlayer<'a, B: Backend> {
    wins: usize,
    wins_as_sente: usize,
    wins_as_gote: usize,
    model: &'a BurnModel<B>,
    device: &'a B::Device,
}

impl<'a, B: Backend> ModelPlayer<'a, B> {
    fn new(model: &'a BurnModel<B>, device: &'a B::Device) -> Self {
        Self {
            wins: 0,
            wins_as_sente: 0,
            wins_as_gote: 0,
            model,
            device,
        }
    }
}

impl<'a, B: Backend> Player for ModelPlayer<'a, B> {
    fn record_win(&mut self, sente: bool) {
        self.wins += 1;
        match sente {
            true => self.wins_as_sente += 1,
            false => self.wins_as_gote += 1,
        }
    }

    fn play(&self, board: Bitboard) -> Bitboard {
        let out = self.model.eval_one(EvalRequest::new(board), self.device);
        let argmax = out
            .policy
            .into_iter()
            .map(Finite::from)
            .argmax()
            .expect("no move");
        board.nth_child(argmax)
    }
}

struct MctsPlayer {
    wins: usize,
    wins_as_sente: usize,
    wins_as_gote: usize,
    sims: u32,
}

impl MctsPlayer {
    fn new(sims: u32) -> Self {
        Self {
            wins: 0,
            wins_as_sente: 0,
            wins_as_gote: 0,
            sims,
        }
    }
}

impl Player for MctsPlayer {
    fn record_win(&mut self, sente: bool) {
        self.wins += 1;
        match sente {
            true => self.wins_as_sente += 1,
            false => self.wins_as_gote += 1,
        }
    }

    fn play(&self, board: Bitboard) -> Bitboard {
        mcts::search(board, board.depth(), MctsSims(self.sims)).best
    }
}

fn play<P1: Player, P2: Player>(sente: &mut P1, gote: &mut P2) -> bool {
    let mut board = Bitboard::new();
    loop {
        if let Some(win) = board.win() {
            match win {
                true => sente.record_win(true),
                false => gote.record_win(false),
            }
            return win;
        }
        board = match board.sente() {
            true => sente.play(board),
            false => gote.play(board),
        };
    }
}

fn make_them_fight<B: Backend>(
    model: &BurnModel<B>,
    device: &B::Device,
    games: usize,
    mcts_sims: u32,
) -> (f64, usize, usize) {
    let mut model_player = ModelPlayer {
        wins: 0,
        wins_as_sente: 0,
        wins_as_gote: 0,
        model,
        device,
    };
    let mut mcts_player = MctsPlayer {
        wins: 0,
        wins_as_sente: 0,
        wins_as_gote: 0,
        sims: mcts_sims,
    };

    for i in (0..games).tqdm() {
        match i.is_multiple_of(2) {
            true => play(&mut model_player, &mut mcts_player),
            false => play(&mut mcts_player, &mut model_player),
        };
    }

    let total = (model_player.wins + mcts_player.wins) as f64;
    let win_rate = model_player.wins as f64 / total;
    (win_rate, model_player.wins_as_sente, model_player.wins_as_gote)
}

trait FloatExt {
    /// Calculate N(mu, sig^2)
    fn normal(self, mu: Self, sig: Self) -> Self;

    /// The logistic function
    fn sigmoid(self) -> Self;

    /// Map (0, 1) to (y0, y1)
    fn lerp(self, y0: Self, y1: Self) -> Self;

    /// Map (y0, y1) to (0, 1)
    fn unlerp(self, y0: Self, y1: Self) -> Self;
}

impl FloatExt for f64 {
    fn normal(self, mu: Self, sig: Self) -> Self {
        let sig2 = 2.0 * sig * sig;
        ((self - mu).powi(2) / -sig2).exp() / (sig2 * PI).sqrt()
    }

    fn sigmoid(self) -> Self {
        1.0 / (1.0 + (-self).exp())
    }

    fn lerp(self, y0: Self, y1: Self) -> Self {
        (1.0 - self) * y0 + self * y1
    }

    fn unlerp(self, y0: Self, y1: Self) -> Self {
        (self - y0) / (y1 - y0)
    }
}
