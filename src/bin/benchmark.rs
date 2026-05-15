use std::{
    f64::consts::PI,
    fs::File,
    io::{self, Write},
    ops::ControlFlow,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, Mutex},
    time::{Duration, Instant},
};

use clap::{Parser, Subcommand};
use hex_table::{
    bb::Bitboard,
    mcts::{self, MctsMonitor, MctsStats},
    nn::{
        candle::model::{CandleDevice, CandleModel},
        search::{Evaluator, search as nn_search},
    },
    util::{Finite, IteratorExt},
};
use rayon::prelude::*;
use tqdm::Iter;

#[derive(Parser, Debug)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Compare(CompareCommand),
    Rank(RankCommand),
    Calibrate(CalibrateCommand),
}

fn main() -> io::Result<()> {
    env_logger::init();

    let cli = Cli::parse();
    log::info!("got options: {cli:?}");

    let device = CandleDevice::default();
    log::info!("device: {device:?}");

    match cli.command {
        Commands::Compare(ref cmd) => cmd_compare(cmd, &device),
        Commands::Rank(ref cmd) => cmd_rank(cmd, &device),
        Commands::Calibrate(ref cmd) => cmd_calibrate(cmd, &device),
    }
}

fn load_checkpoint(path: &Path, device: &CandleDevice) -> io::Result<CandleModel> {
    let bytes = std::fs::read(path)?;
    CandleModel::load_burn(&bytes, device).map_err(io::Error::other)
}

fn list_checkpoints(model_dir: &Path) -> io::Result<Vec<String>> {
    let mut checkpoints = std::fs::read_dir(model_dir)?
        .filter_map(|x| x.ok())
        .flat_map(|x| x.file_name().into_string().into_iter())
        .filter(|x| x.starts_with("checkpoint-"))
        .collect::<Vec<_>>();
    checkpoints.sort();
    log::info!("found {} checkpoints in {}", checkpoints.len(), model_dir.display());
    Ok(checkpoints)
}

/// Compare model checkpoints to an MCTS benchmark player
#[derive(Parser, Debug)]
struct CompareCommand {
    /// The model directory
    #[arg(long, value_name = "DIR")]
    model_dir: PathBuf,

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

fn cmd_compare(cmd: &CompareCommand, device: &CandleDevice) -> io::Result<()> {
    let mut checkpoints = list_checkpoints(&cmd.model_dir)?;

    if let Some(n) = cmd.checkpoints
        && n < checkpoints.len()
    {
        log::info!("using only {n} most recent checkpoints");
        let _ = checkpoints.drain(..checkpoints.len() - n);
    }

    let first = checkpoints
        .first()
        .ok_or_else(|| io::Error::other("no checkpoints"))?;
    let nn_evals_per_time = {
        let model = load_checkpoint(&cmd.model_dir.join(first), device)?;
        bench_model_evals(&model, false)
    };
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
        let model = load_checkpoint(&cmd.model_dir.join(checkpoint), device)?;
        let (win_rate, wins_as_sente, wins_as_gote) =
            make_them_fight(&model, cmd.games, mcts_per_nn);
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

/// Match a subject strategy's performance against a baseline strategy whose
/// strength is varied via Bayesian inference until the win rate balances.
///
/// Baseline spec: `mcts:LO-HI` or `model:LO-HI:PATH` (range defines the prior
/// support; for mcts, `10^rank` rollouts; for model, `10^rank` nn-search iters).
/// Subject spec: `mcts:RANK` or `model:RANK:GLOB` (fixed rank; for model, the
/// glob expands to one CSV row per matched checkpoint; `model:0` is intuition).
#[derive(Parser, Debug)]
struct RankCommand {
    /// Baseline strategy spec
    #[arg(long, value_name = "SPEC", value_parser = parse_baseline)]
    baseline: BaselineSpec,

    /// Subject strategy spec
    #[arg(long, value_name = "SPEC", value_parser = parse_subject)]
    subject: SubjectSpec,

    /// CSV file to write results to. If omitted, no CSV is written.
    #[arg(long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Stop at the given stddev cutoff
    #[arg(long, value_name = "X")]
    stddev_stop: Option<f64>,

    /// Stop at the given number of iterations
    #[arg(long, value_name = "N")]
    iters_stop: Option<usize>,

    /// Win-probability coefficients B0,B1 such that
    /// P(sente_win) = sigmoid(B0 + B1 * (sente_rank - gote_rank)).
    /// Defaults are calibrated from skill.ipynb; refit with
    /// `benchmark calibrate` if they go stale.
    #[arg(long, value_name = "B0,B1", default_value = "0.3,1.8", value_parser = parse_likelihood)]
    likelihood: Likelihood,
}

#[derive(Clone, Debug)]
enum BaselineSpec {
    Mcts { lo: f64, hi: f64 },
    Model { lo: f64, hi: f64, path: PathBuf },
}

#[derive(Clone, Debug)]
enum SubjectSpec {
    Mcts { rank: f64 },
    Model { rank: f64, glob: String },
}

impl BaselineSpec {
    fn descriptor(&self) -> String {
        match self {
            BaselineSpec::Mcts { lo, hi } => format!("mcts:{lo}-{hi}"),
            BaselineSpec::Model { lo, hi, path } => {
                format!("model:{lo}-{hi}:{}", path.display())
            }
        }
    }
}

fn parse_range(s: &str) -> Result<(f64, f64), String> {
    let (lo, hi) = s
        .split_once('-')
        .ok_or_else(|| format!("expected LO-HI, got {s:?}"))?;
    let lo: f64 = lo.parse().map_err(|e| format!("bad LO {lo:?}: {e}"))?;
    let hi: f64 = hi.parse().map_err(|e| format!("bad HI {hi:?}: {e}"))?;
    if !(lo < hi) {
        return Err(format!("expected LO < HI, got {lo}-{hi}"));
    }
    Ok((lo, hi))
}

fn parse_baseline(s: &str) -> Result<BaselineSpec, String> {
    let (kind, rest) = s
        .split_once(':')
        .ok_or_else(|| format!("expected KIND:..., got {s:?}"))?;
    match kind {
        "mcts" => {
            let (lo, hi) = parse_range(rest)?;
            Ok(BaselineSpec::Mcts { lo, hi })
        }
        "model" => {
            let (range, path) = rest
                .split_once(':')
                .ok_or_else(|| format!("expected model:LO-HI:PATH, got {s:?}"))?;
            let (lo, hi) = parse_range(range)?;
            Ok(BaselineSpec::Model {
                lo,
                hi,
                path: PathBuf::from(path),
            })
        }
        _ => Err(format!("unknown baseline kind {kind:?} (expected mcts or model)")),
    }
}

fn parse_subject(s: &str) -> Result<SubjectSpec, String> {
    let (kind, rest) = s
        .split_once(':')
        .ok_or_else(|| format!("expected KIND:..., got {s:?}"))?;
    match kind {
        "mcts" => {
            let rank: f64 = rest
                .parse()
                .map_err(|e| format!("bad rank {rest:?}: {e}"))?;
            Ok(SubjectSpec::Mcts { rank })
        }
        "model" => {
            let (rank, glob) = rest
                .split_once(':')
                .ok_or_else(|| format!("expected model:RANK:GLOB, got {s:?}"))?;
            let rank: f64 = rank
                .parse()
                .map_err(|e| format!("bad rank {rank:?}: {e}"))?;
            Ok(SubjectSpec::Model {
                rank,
                glob: glob.to_string(),
            })
        }
        _ => Err(format!("unknown subject kind {kind:?} (expected mcts or model)")),
    }
}

/// Coefficients for the win-probability sigmoid:
/// `P(sente_win) = sigmoid(b0 + b1 * (sente_rank - gote_rank))`.
#[derive(Copy, Clone, Debug)]
struct Likelihood {
    b0: f64,
    b1: f64,
}

fn parse_likelihood(s: &str) -> Result<Likelihood, String> {
    let (b0, b1) = s
        .split_once(',')
        .ok_or_else(|| format!("expected B0,B1, got {s:?}"))?;
    let b0: f64 = b0.parse().map_err(|e| format!("bad B0 {b0:?}: {e}"))?;
    let b1: f64 = b1.parse().map_err(|e| format!("bad B1 {b1:?}: {e}"))?;
    Ok(Likelihood { b0, b1 })
}

/// A resolved player kind: an MCTS searcher or a loaded model. Wraps the only
/// runtime state that distinguishes a `mcts:` from a `model:` spec, so
/// downstream code can dispatch through [`Self::player_at`] without
/// re-matching on the parse-time spec or threading an `Option<CandleModel>`.
enum Strategy {
    Mcts,
    Model(CandleModel),
}

impl Strategy {
    /// Build a player for one game at the given rank. For `model:` strategies,
    /// `rank == 0` is treated as pure intuition (no nn-search); otherwise the
    /// player runs `10^rank` MCTS / nn-search iterations.
    fn player_at(&self, rank: f64) -> Box<dyn Player + '_> {
        match self {
            Strategy::Mcts => Box::new(MctsPlayer::new(10.0f64.powf(rank) as u32)),
            Strategy::Model(m) => {
                let sims = 10.0f64.powf(rank) as u32;
                Box::new(ModelPlayer::new(m, sims))
            }
        }
    }

    fn bench_bps(&self, fast: bool) -> f64 {
        match self {
            Strategy::Mcts => bench_mcts_evals(fast),
            Strategy::Model(m) => bench_model_evals(m, fast),
        }
    }
}

/// A baseline whose range and (optional) model have been resolved from a
/// [`BaselineSpec`]. Owned for the lifetime of one command invocation.
struct ResolvedBaseline {
    strategy: Strategy,
    lo: f64,
    hi: f64,
    descriptor: String,
}

impl ResolvedBaseline {
    fn resolve(spec: &BaselineSpec, device: &CandleDevice) -> io::Result<Self> {
        let descriptor = spec.descriptor();
        let (strategy, lo, hi) = match spec {
            BaselineSpec::Mcts { lo, hi } => (Strategy::Mcts, *lo, *hi),
            BaselineSpec::Model { lo, hi, path } => {
                (Strategy::Model(load_checkpoint(path, device)?), *lo, *hi)
            }
        };
        Ok(Self {
            strategy,
            lo,
            hi,
            descriptor,
        })
    }
}

/// One subject in the comparison: a strategy at a fixed rank. For `mcts:`
/// subjects there is exactly one; for `model:` subjects there is one per
/// glob match.
struct ResolvedSubject {
    strategy: Strategy,
    rank: f64,
    descriptor: String,
}

impl ResolvedSubject {
    fn expand(spec: &SubjectSpec, device: &CandleDevice) -> io::Result<Vec<Self>> {
        match spec {
            SubjectSpec::Mcts { rank } => Ok(vec![Self {
                strategy: Strategy::Mcts,
                rank: *rank,
                descriptor: format!("mcts:{rank}"),
            }]),
            SubjectSpec::Model {
                rank,
                glob: pattern,
            } => {
                let paths: Vec<PathBuf> = glob::glob(pattern)
                    .map_err(|e| io::Error::other(format!("bad glob {pattern:?}: {e}")))?
                    .filter_map(|r| r.ok())
                    .collect();
                if paths.is_empty() {
                    return Err(io::Error::other(format!(
                        "subject glob matched no files: {pattern}"
                    )));
                }
                paths
                    .into_iter()
                    .map(|p| -> io::Result<Self> {
                        let descriptor = format!("model:{rank}:{}", p.display());
                        let model = load_checkpoint(&p, device)?;
                        Ok(Self {
                            strategy: Strategy::Model(model),
                            rank: *rank,
                            descriptor,
                        })
                    })
                    .collect()
            }
        }
    }
}

fn cmd_rank(cmd: &RankCommand, device: &CandleDevice) -> io::Result<()> {
    let baseline = ResolvedBaseline::resolve(&cmd.baseline, device)?;
    let subjects = ResolvedSubject::expand(&cmd.subject, device)?;
    log::info!("expanded subject into {} point(s)", subjects.len());

    let baseline_bps = baseline.strategy.bench_bps(true);
    // All subjects share the same strategy kind (they came from one
    // SubjectSpec); for `model:` subjects we also assume all checkpoints share
    // the same architecture, so benching the first stands in for the rest.
    let subject_bps = subjects[0].strategy.bench_bps(true);

    let mut out = cmd
        .output
        .as_ref()
        .map(|p| {
            std::fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(p)
        })
        .transpose()?;
    if let Some(out) = out.as_mut() {
        writeln!(
            out,
            "baseline,subject,rank,compute_equiv_rank,mean,stddev,iters,elapsed_seconds"
        )?;
        out.flush().ok();
    }

    let stddev_stop = cmd.stddev_stop;
    let iters_stop = match (cmd.iters_stop, cmd.stddev_stop) {
        (None, None) => {
            log::warn!("using default stopping condition of 40 iters");
            Some(40)
        }
        (iters, _) => iters,
    };

    for subject in &subjects {
        let compute_equiv_rank =
            ((subject.rank + (baseline_bps / subject_bps).log10()) * 100.0).round() / 100.0;
        rank_subject(
            out.as_mut(),
            &baseline,
            subject,
            compute_equiv_rank,
            iters_stop,
            stddev_stop,
            &cmd.likelihood,
        )?;
    }

    Ok(())
}

fn rank_subject(
    mut out: Option<&mut File>,
    baseline: &ResolvedBaseline,
    subject: &ResolvedSubject,
    compute_equiv_rank: f64,
    iters_stop: Option<usize>,
    stddev_stop: Option<f64>,
    likelihood: &Likelihood,
) -> io::Result<()> {
    const RANK_SUBUNITS: f64 = 128.0;

    let ranks_n = ((baseline.hi - baseline.lo) * RANK_SUBUNITS + 1.0).round() as usize;
    let ranks_xs = Linspace::new(baseline.lo, baseline.hi, ranks_n);
    let mut ranks = Prior::from_fn(ranks_xs, |x| {
        let range = baseline.hi - baseline.lo;
        let mid = baseline.lo + range / 2.0;
        let uniform = 1.0 / (range) as f64;
        let normal = x.normal(mid, range / 2.0);
        0.75.lerp(normal, uniform)
    });

    log::info!("ranking subject {} against baseline {}", subject.descriptor, baseline.descriptor,);

    let start = Instant::now();

    for iter in 0usize.. {
        let rank = ranks.argmax();
        let stats = ranks.stats();

        let handicap = rank - compute_equiv_rank;
        let stddev = stats.variance.sqrt();

        ranks.show(1.0);
        log::info!(
            "mean={:.2} stddev={:.2} rank={:.2} handicap={:.2}",
            stats.mean,
            stddev,
            rank,
            handicap,
        );

        let stop = iters_stop.map(|x| x <= iter).unwrap_or(false)
            || stddev_stop.map(|x| stddev <= x).unwrap_or(false);
        if stop {
            let lo = 10.0f64.powf(handicap - stddev * 2.0);
            let hi = 10.0f64.powf(handicap + stddev * 2.0);
            log::info!("subject seems {lo:.2}x-{hi:.2}x baseline strength");
            if let Some(out) = out.as_deref_mut() {
                writeln!(
                    out,
                    "{},{},{:.2},{:.2},{:.5},{:.5},{},{}",
                    baseline.descriptor,
                    subject.descriptor,
                    rank,
                    compute_equiv_rank,
                    stats.mean,
                    stddev,
                    iter,
                    start.elapsed().as_secs_f64(),
                )?;
                out.flush().ok();
            }
            break;
        }

        let subject_is_sente = iter.is_multiple_of(2);
        let mut subject_player = subject.strategy.player_at(subject.rank);
        let mut baseline_player = baseline.strategy.player_at(rank);
        let sente_win = play_one(&mut *subject_player, &mut *baseline_player, subject_is_sente);

        ranks.update(|x| {
            let sente_relative_rank = if subject_is_sente { x - rank } else { rank - x };
            let p = p_sente_win(likelihood, sente_relative_rank);
            match sente_win {
                true => p,
                false => 1.0 - p,
            }
        });
    }

    Ok(())
}

/// Runs `play(sente, gote)` but lets the caller think in terms of who is the
/// subject vs the baseline. Returns whether sente won.
fn play_one(subject: &mut dyn Player, baseline: &mut dyn Player, subject_is_sente: bool) -> bool {
    if subject_is_sente { play(subject, baseline, false) } else { play(baseline, subject, true) }
}

/// Sample random baseline-rank pairs, play one game per pair, and fit a
/// logistic regression on the win/loss outcomes. The output is a `(b0, b1)`
/// pair such that `P(sente_win) = sigmoid(b0 + b1 * (s_rank - g_rank))`.
#[derive(Parser, Debug)]
struct CalibrateCommand {
    /// Baseline strategy spec (defines the rank range sampled from)
    #[arg(long, value_name = "SPEC", value_parser = parse_baseline)]
    baseline: BaselineSpec,

    /// Number of games to play
    #[arg(long, value_name = "N", default_value = "1000")]
    games: usize,

    /// CSV file to write per-game results to. If omitted, no CSV is written.
    #[arg(long, value_name = "FILE")]
    output: Option<PathBuf>,
}

fn cmd_calibrate(cmd: &CalibrateCommand, device: &CandleDevice) -> io::Result<()> {
    let baseline = ResolvedBaseline::resolve(&cmd.baseline, device)?;
    log::info!(
        "calibrating {}: {} games, rank ∈ [{}, {}]",
        baseline.descriptor,
        cmd.games,
        baseline.lo,
        baseline.hi,
    );

    let out_file = cmd
        .output
        .as_ref()
        .map(|p| {
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(p)?;
            writeln!(f, "baseline,sente_rank,gote_rank,sente_win,turns")?;
            f.flush().ok();
            io::Result::Ok(f)
        })
        .transpose()?;
    let out = Arc::new(Mutex::new(out_file));

    let pbar = Arc::new(Mutex::new(tqdm::pbar(Some(cmd.games))));

    // The candle Metal backend isn't safe for concurrent forward passes (it
    // can corrupt internal state and produce NaN policies — and the
    // CandleModel guard will panic on contended access). For model:
    // baselines we serialize games into a single-threaded rayon pool. For
    // mcts: baselines we let rayon use its default thread count.
    let n_threads = match baseline.strategy {
        Strategy::Mcts => 0,
        Strategy::Model(_) => 1,
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(io::Error::other)?;

    let rows: Vec<(f64, f64, bool)> = pool.install(|| {
        (0..cmd.games)
            .into_par_iter()
            .map(|_| {
                let s_rank: f64 = rand::random_range(baseline.lo..=baseline.hi);
                let g_rank: f64 = rand::random_range(baseline.lo..=baseline.hi);
                let (sente_win, turns) = play_calibrate_game(&baseline.strategy, s_rank, g_rank);
                if let Some(out) = out.lock().unwrap().as_mut() {
                    writeln!(
                        out,
                        "{},{s_rank},{g_rank},{},{}",
                        baseline.descriptor, sente_win as u8, turns,
                    )
                    .ok();
                    out.flush().ok();
                }
                pbar.lock().unwrap().update(1).ok();
                (s_rank, g_rank, sente_win)
            })
            .collect()
    });
    pbar.lock().unwrap().close().ok();

    let xs: Vec<f64> = rows.iter().map(|(s, g, _)| s - g).collect();
    let ys: Vec<f64> = rows.iter().map(|(_, _, w)| *w as u8 as f64).collect();
    let (b0, b1) = fit_logistic(&xs, &ys);
    log::info!("logistic regression: P(sente_win) = sigmoid({b0:.4} + {b1:.4} * rank_diff)");
    println!();
    println!("p_sente_win(diff) = sigmoid({b0:.4} + {b1:.4} * diff)");
    println!("  b0 = {b0:.6}");
    println!("  b1 = {b1:.6}");

    Ok(())
}

fn play_calibrate_game(strategy: &Strategy, s_rank: f64, g_rank: f64) -> (bool, usize) {
    let sente = strategy.player_at(s_rank);
    let gote = strategy.player_at(g_rank);
    let mut board = Bitboard::new();
    for turn in 0.. {
        if let Some(win) = board.win() {
            return (win, turn);
        }
        board = match board.sente() {
            true => sente.play(board),
            false => gote.play(board),
        };
    }
    unreachable!()
}

/// Newton-Raphson fit of a 2-parameter logistic regression:
/// `P(y=1 | x) = sigmoid(b0 + b1 * x)`. Returns `(b0, b1)`. Unregularized
/// (maximum-likelihood); converges in ~5-10 iterations for typical data.
fn fit_logistic(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    assert_eq!(xs.len(), ys.len());
    let mut b0 = 0.0;
    let mut b1 = 0.0;
    for _ in 0..100 {
        let mut g0 = 0.0;
        let mut g1 = 0.0;
        let mut h00 = 0.0;
        let mut h01 = 0.0;
        let mut h11 = 0.0;
        for i in 0..xs.len() {
            let z = b0 + b1 * xs[i];
            let p = 1.0 / (1.0 + (-z).exp());
            let err = p - ys[i];
            g0 += err;
            g1 += err * xs[i];
            let w = p * (1.0 - p);
            h00 += w;
            h01 += w * xs[i];
            h11 += w * xs[i] * xs[i];
        }
        let det = h00 * h11 - h01 * h01;
        if det.abs() < 1e-12 {
            break;
        }
        let d0 = (h11 * g0 - h01 * g1) / det;
        let d1 = (-h01 * g0 + h00 * g1) / det;
        b0 -= d0;
        b1 -= d1;
        if d0.abs() < 1e-8 && d1.abs() < 1e-8 {
            break;
        }
    }
    (b0, b1)
}

fn p_sente_win(likelihood: &Likelihood, sente_relative_rank: f64) -> f64 {
    (likelihood.b0 + likelihood.b1 * sente_relative_rank).sigmoid()
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

fn bench_model_evals(model: &CandleModel, fast: bool) -> f64 {
    let board = Bitboard::new();

    log::info!("benchmarking model evals");

    let dur = match fast {
        true => Duration::from_secs(1),
        false => Duration::from_secs(30),
    };
    log::info!("warming up for {dur:?}");
    let start = Instant::now();
    while start.elapsed() < dur {
        std::hint::black_box(model.call(board));
    }

    let dur = match fast {
        true => Duration::from_secs(1),
        false => Duration::from_secs(5),
    };
    log::info!("counting evals in {dur:?}");
    let mut count = 0;
    let start = Instant::now();
    while start.elapsed() < dur {
        std::hint::black_box(model.call(board));
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

struct ModelPlayer<'a> {
    wins: usize,
    wins_as_sente: usize,
    wins_as_gote: usize,
    model: &'a CandleModel,
    /// Number of nn-MCTS iterations per move. `0` means pure intuition: take
    /// the policy argmax with no tree search.
    sims: u32,
}

impl<'a> ModelPlayer<'a> {
    fn new(model: &'a CandleModel, sims: u32) -> Self {
        Self {
            wins: 0,
            wins_as_sente: 0,
            wins_as_gote: 0,
            model,
            sims,
        }
    }
}

impl<'a> Player for ModelPlayer<'a> {
    fn record_win(&mut self, sente: bool) {
        self.wins += 1;
        match sente {
            true => self.wins_as_sente += 1,
            false => self.wins_as_gote += 1,
        }
    }

    fn play(&self, board: Bitboard) -> Bitboard {
        // we use sample here to inject some randomness into the rating system
        if self.sims <= 2 {
            let out = self.model.call(board);
            let sample = out
                .policy
                .iter()
                .enumerate()
                .map(|(i, x)| if board.nth_child_valid(i) { *x } else { 0.0 })
                .sample_weighted(&mut rand::rng())
                .expect("no move");
            board.nth_child(sample)
        } else {
            let target = self.sims as usize;
            nn_search(self.model, board, 0.0, 0.0, |n: usize| n < target).board_sample
        }
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

fn play(sente: &mut dyn Player, gote: &mut dyn Player, transpose: bool) -> bool {
    let mut board = Bitboard::new();
    loop {
        if let Some(win) = board.win() {
            println!(
                "{}",
                hex_table::bb::BitboardPretty(&if transpose { board.transpose() } else { board })
            );
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

fn make_them_fight(model: &CandleModel, games: usize, mcts_sims: u32) -> (f64, usize, usize) {
    let mut model_player = ModelPlayer {
        wins: 0,
        wins_as_sente: 0,
        wins_as_gote: 0,
        model,
        sims: 0,
    };
    let mut mcts_player = MctsPlayer {
        wins: 0,
        wins_as_sente: 0,
        wins_as_gote: 0,
        sims: mcts_sims,
    };

    for i in (0..games).tqdm() {
        match i.is_multiple_of(2) {
            true => play(&mut model_player, &mut mcts_player, false),
            false => play(&mut mcts_player, &mut model_player, true),
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
