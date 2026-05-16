use std::{
    f64::consts::PI,
    fs::File,
    io::{self, Write},
    ops::ControlFlow,
    path::{Path, PathBuf},
    sync::LazyLock,
    time::{Duration, Instant},
};

use clap::{Parser, Subcommand};
use hex_table::{
    bb::{Bitboard, ExactMcts},
    mcts::{self, MctsMonitor, MctsStats},
    nn::{
        candle::model::{CandleDevice, CandleModel},
        search::search as nn_search,
    },
    util::{Finite, IteratorExt},
};

#[derive(Parser, Debug)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Rank(RankCommand),
}

fn main() -> io::Result<()> {
    env_logger::init();

    let cli = Cli::parse();
    log::info!("got options: {cli:?}");

    let device = CandleDevice::default();
    log::info!("device: {device:?}");

    match cli.command {
        Commands::Rank(ref cmd) => cmd_rank(cmd, &device),
    }
}

fn load_checkpoint(path: &Path, device: &CandleDevice) -> io::Result<CandleModel> {
    let bytes = std::fs::read(path)?;
    CandleModel::load_burn(&bytes, device).map_err(io::Error::other)
}

// ============================================================================
// rank
// ============================================================================

/// Bayesian estimate of the compute-time handicap at which a fixed `subject`
/// strategy ties with a `baseline` strategy.
///
/// The prior is over `handicap = log10(baseline_time / subject_time)` — i.e.,
/// the log compute-time factor the baseline gets relative to the subject.
/// Each iteration plays one game with per-turn time budgets derived
/// symmetrically from the current argmax handicap, observes the effective
/// handicap from the actual time each side spent, and updates the prior with
/// the observed handicap (not the suggested one).
#[derive(Parser, Debug)]
struct RankCommand {
    /// Baseline checkpoint path. Omit for an MCTS baseline.
    #[arg(long, value_name = "PATH")]
    baseline: Option<PathBuf>,

    /// Subject checkpoint paths (one row per path). Omit for a single MCTS
    /// subject; pass multiple to rank a whole set — shell-expand globs.
    #[arg(long, value_name = "PATH", num_args = 1..)]
    subject: Vec<PathBuf>,

    /// CSV file to write results to. If omitted, no CSV is written.
    #[arg(long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Stop at the given stddev cutoff.
    #[arg(long, value_name = "X")]
    stddev_stop: Option<f64>,

    /// Stop at the given number of iterations.
    #[arg(long, value_name = "N")]
    iters_stop: Option<usize>,

    /// Lower bound of the handicap prior.
    #[arg(
        long,
        value_name = "X",
        default_value = "-3.0",
        allow_hyphen_values = true
    )]
    prior_min: f64,

    /// Upper bound of the handicap prior.
    #[arg(
        long,
        value_name = "X",
        default_value = "3.0",
        allow_hyphen_values = true
    )]
    prior_max: f64,

    /// Per-turn base time, in seconds. Subject and baseline each get
    /// `base_time * 10^(±handicap/2)` per turn, floored at 100 ms.
    #[arg(long, value_name = "SECONDS", default_value = "1.0")]
    base_time: f64,

    /// Win-probability coefficients B0,B1 such that
    /// P(sente_win) = sigmoid(B0 + B1 * sente_advantage), where
    /// sente_advantage = log10(sente_total_time / gote_total_time). Defaults
    /// are stale from the old sim-based calibration.
    #[arg(long, value_name = "B0,B1", default_value = "0.3,1.8", value_parser = parse_likelihood)]
    likelihood: Likelihood,
}

/// A baseline or subject in the matchup: an MCTS searcher or a loaded model.
/// `play()` runs one move respecting the deadline.
struct Strategy {
    model: Option<CandleModel>,
    exact: bool,
    descriptor: String,
}

impl Strategy {
    fn resolve(path: Option<&Path>, device: &CandleDevice) -> io::Result<Self> {
        match path {
            None => Ok(Self {
                model: None,
                exact: false,
                descriptor: "mcts".to_string(),
            }),
            Some(p) if p.to_str() == Some("mcts-exact") => Ok(Self {
                model: None,
                exact: true,
                descriptor: "mcts-exact".to_string(),
            }),
            Some(p) => Ok(Self {
                model: Some(load_checkpoint(p, device)?),
                exact: false,
                descriptor: format!("model:{}", p.display()),
            }),
        }
    }

    fn play(&self, board: Bitboard, budget: Duration) -> Bitboard {
        let start = Instant::now();
        match &self.model {
            None => {
                let deadline = start + budget;
                if self.exact {
                    mcts::search(ExactMcts(board), board.depth(), MctsDeadline(deadline))
                        .best
                        .0
                } else {
                    mcts::search(board, board.depth(), MctsDeadline(deadline)).best
                }
            }
            Some(m) => {
                let deadline = start + budget.max(Duration::from_millis(100));
                let out = nn_search(m, board, 0.0, 0.0, move |_n: usize| {
                    Instant::now() < deadline
                });
                let i = out
                    .policy
                    .iter()
                    .copied()
                    .map(|x| x.powi(4))
                    .sample_weighted(&mut rand::rng())
                    .unwrap();
                board.nth_child(i)
            }
        }
    }
}

fn expand_subjects(paths: &[PathBuf], device: &CandleDevice) -> io::Result<Vec<Strategy>> {
    if paths.is_empty() {
        Ok(vec![Strategy::resolve(None, device)?])
    } else {
        paths
            .iter()
            .map(|p| Strategy::resolve(Some(p), device))
            .collect()
    }
}

/// Per-side compute time and outcome of one game.
struct GameOutcome {
    sente_win: bool,
    sente_total: Duration,
    gote_total: Duration,
}

const MIN_BUDGET: Duration = Duration::from_millis(1);

/// Split a suggested handicap into per-turn budgets, symmetrically around
/// `base_time`. Both budgets are floored at 100 ms, so the *effective*
/// handicap of the game may differ from the suggestion when a side hits the
/// floor — the rank update uses what actually got played, not what was asked.
/// Returns `(subject_budget, baseline_budget)`.
fn budgets_from_handicap(suggested: f64, base_time: f64) -> (Duration, Duration) {
    let subject_secs = base_time * 10.0f64.powf(-suggested / 2.0);
    let baseline_secs = base_time * 10.0f64.powf(suggested / 2.0);
    let subject = Duration::from_secs_f64(subject_secs).max(MIN_BUDGET);
    let baseline = Duration::from_secs_f64(baseline_secs).max(MIN_BUDGET);
    (subject, baseline)
}

fn play_game(
    sente: &Strategy,
    gote: &Strategy,
    sente_budget: Duration,
    gote_budget: Duration,
) -> GameOutcome {
    let mut board = Bitboard::new();
    let mut sente_total = Duration::ZERO;
    let mut gote_total = Duration::ZERO;
    loop {
        show_game(&board);
        if let Some(win) = board.win() {
            return GameOutcome {
                sente_win: win,
                sente_total,
                gote_total,
            };
        }
        let is_sente = board.sente();
        let (strategy, budget) = if is_sente {
            (sente, sente_budget)
        } else {
            (gote, gote_budget)
        };
        let start = Instant::now();
        let next = strategy.play(board, budget);
        let elapsed = start.elapsed();
        if is_sente {
            sente_total += elapsed;
        } else {
            gote_total += elapsed;
        }
        board = next;
    }
}

fn show_game(board: &Bitboard) {
    use hex_table::bb::BitboardPretty;
    print!("\x1b[s");
    print!("\x1b[1;1H");
    for _ in 0..13 {
        println!("\x1b[K");
    }
    print!("\x1b[1;1H{}", BitboardPretty(board));
    print!("\x1b[u");
    std::io::stdout().flush().ok();
}

fn cmd_rank(cmd: &RankCommand, device: &CandleDevice) -> io::Result<()> {
    let baseline = Strategy::resolve(cmd.baseline.as_deref(), device)?;
    let subjects = expand_subjects(&cmd.subject, device)?;
    log::info!(
        "baseline {}; {} subject(s)",
        baseline.descriptor,
        subjects.len()
    );

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
            "baseline,subject,handicap,mean,stddev,iters,elapsed_seconds"
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
        rank_subject(
            out.as_mut(),
            &baseline,
            subject,
            cmd.prior_min,
            cmd.prior_max,
            cmd.base_time,
            iters_stop,
            stddev_stop,
            &cmd.likelihood,
        )?;
    }

    Ok(())
}

fn rank_subject(
    mut out: Option<&mut File>,
    baseline: &Strategy,
    subject: &Strategy,
    prior_min: f64,
    prior_max: f64,
    base_time: f64,
    iters_stop: Option<usize>,
    stddev_stop: Option<f64>,
    likelihood: &Likelihood,
) -> io::Result<()> {
    const RANK_SUBUNITS: f64 = 128.0;

    let ranks_n = ((prior_max - prior_min) * RANK_SUBUNITS + 1.0).round() as usize;
    let ranks_xs = Linspace::new(prior_min, prior_max, ranks_n);
    // Mild normal at handicap = 0 (equal compute), mixed with uniform so the
    // tails never go to zero.
    let mut ranks = Prior::from_fn(ranks_xs, |x| {
        let uniform = 1.0 / ranks_n as f64;
        let normal = x.normal(0.0, 2.0) / RANK_SUBUNITS;
        0.75.lerp(normal, uniform)
    });

    log::info!(
        "ranking subject {} against baseline {}",
        subject.descriptor,
        baseline.descriptor,
    );

    let start = Instant::now();

    for iter in 0usize.. {
        let handicap = ranks.argmax();
        let stats = ranks.stats();
        let stddev = stats.variance.sqrt();

        ranks.show(1.0);
        log::info!(
            "iter={iter} mean={:.2} stddev={:.2} handicap={:.2}",
            stats.mean,
            stddev,
            handicap,
        );

        let stop = iters_stop.map(|x| x <= iter).unwrap_or(false)
            || stddev_stop.map(|x| stddev <= x).unwrap_or(false);
        if stop {
            log::info!(
                "subject seems balanced with baseline at handicap {:.2} (95% CI ±{:.2})",
                handicap,
                stddev * 2.0,
            );
            if let Some(out) = out.as_deref_mut() {
                writeln!(
                    out,
                    "{},{},{:.2},{:.5},{:.5},{},{}",
                    baseline.descriptor,
                    subject.descriptor,
                    handicap,
                    stats.mean,
                    stddev,
                    iter,
                    start.elapsed().as_secs_f64(),
                )?;
                out.flush().ok();
            }
            break;
        }

        let (subject_budget, baseline_budget) = budgets_from_handicap(handicap, base_time);
        let subject_is_sente = iter.is_multiple_of(2);
        let (sente, gote, sente_budget, gote_budget) = if subject_is_sente {
            (subject, baseline, subject_budget, baseline_budget)
        } else {
            (baseline, subject, baseline_budget, subject_budget)
        };
        let outcome = play_game(sente, gote, sente_budget, gote_budget);

        // Effective handicap from the baseline's perspective: positive means
        // baseline actually got more compute than the subject.
        let baseline_secs = if subject_is_sente {
            outcome.gote_total.as_secs_f64()
        } else {
            outcome.sente_total.as_secs_f64()
        };
        let subject_secs = if subject_is_sente {
            outcome.sente_total.as_secs_f64()
        } else {
            outcome.gote_total.as_secs_f64()
        };
        let h_eff = (baseline_secs / subject_secs).log10();

        log::info!(
            "  subject_is_sente={subject_is_sente} h_eff={h_eff:.2} sente_win={}",
            outcome.sente_win,
        );

        ranks.update(|x| {
            // `x` is a candidate balanced handicap (i.e., the handicap at
            // which the matchup is even). The baseline's actual advantage in
            // this game over the balanced point is (h_eff - x), translating
            // into a signed sente_advantage that depends on who was sente.
            let sente_advantage = if subject_is_sente { x - h_eff } else { h_eff - x };
            let p = p_sente_win(likelihood, sente_advantage);
            if outcome.sente_win { p } else { 1.0 - p }
        });
    }

    Ok(())
}

/// MCTS monitor that runs the search until [`Instant::now`] reaches the
/// deadline.
struct MctsDeadline(Instant);

impl<S> MctsMonitor<S> for MctsDeadline {
    fn defer(&mut self, _stats: &MctsStats<S>) -> ControlFlow<()> {
        use ControlFlow::*;
        if Instant::now() < self.0 {
            Continue(())
        } else {
            Break(())
        }
    }
}

// ============================================================================
// p_sente_win and likelihood
// ============================================================================

fn p_sente_win(likelihood: &Likelihood, sente_advantage: f64) -> f64 {
    (likelihood.b0 + likelihood.b1 * sente_advantage).sigmoid()
}

/// Coefficients for the win-probability sigmoid:
/// `P(sente_win) = sigmoid(b0 + b1 * sente_advantage)`.
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

// ============================================================================
// Linspace, Prior — discrete prior on a 1D grid
// ============================================================================

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

/// A discrete prior distribution.
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

// ============================================================================
// FloatExt
// ============================================================================

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

// ============================================================================
// calibrate (commented out pending redesign for time-based players)
// ============================================================================

/*

// The calibrate subcommand collected random rank pairs of MCTS-vs-MCTS games
// and fit a logistic regression to derive the (b0, b1) coefficients used by
// p_sente_win. The rank-based formulation is obsolete — the new design wants
// the regression in terms of log10(time-ratio) rather than rank diff. Left
// here as a reference until we redesign it.

#[derive(Parser, Debug)]
struct CalibrateCommand {
    #[arg(long, value_name = "SPEC", value_parser = parse_baseline)]
    baseline: BaselineSpec,
    #[arg(long, value_name = "N", default_value = "1000")]
    games: usize,
    #[arg(long, value_name = "FILE")]
    output: Option<PathBuf>,
}

fn cmd_calibrate(cmd: &CalibrateCommand, device: &CandleDevice) -> io::Result<()> {
    let baseline = ResolvedBaseline::resolve(&cmd.baseline, device)?;
    log::info!(
        "calibrating {}: {} games, rank ∈ [{}, {}]",
        baseline.descriptor, cmd.games, baseline.lo, baseline.hi,
    );

    let out_file = cmd.output.as_ref().map(|p| {
        let mut f = std::fs::OpenOptions::new().create(true).truncate(true).write(true).open(p)?;
        writeln!(f, "baseline,sente_rank,gote_rank,sente_win,turns")?;
        f.flush().ok();
        io::Result::Ok(f)
    }).transpose()?;
    let out = Arc::new(Mutex::new(out_file));

    let pbar = Arc::new(Mutex::new(tqdm::pbar(Some(cmd.games))));
    let n_threads = match baseline.strategy {
        Strategy::Mcts => 0,
        Strategy::Model(_) => 1,
    };
    let pool = rayon::ThreadPoolBuilder::new().num_threads(n_threads).build().map_err(io::Error::other)?;

    let rows: Vec<(f64, f64, bool)> = pool.install(|| {
        (0..cmd.games).into_par_iter().map(|_| {
            let s_rank: f64 = rand::random_range(baseline.lo..=baseline.hi);
            let g_rank: f64 = rand::random_range(baseline.lo..=baseline.hi);
            let (sente_win, turns) = play_calibrate_game(&baseline.strategy, s_rank, g_rank);
            if let Some(out) = out.lock().unwrap().as_mut() {
                writeln!(out, "{},{s_rank},{g_rank},{},{}", baseline.descriptor, sente_win as u8, turns).ok();
                out.flush().ok();
            }
            pbar.lock().unwrap().update(1).ok();
            (s_rank, g_rank, sente_win)
        }).collect()
    });
    pbar.lock().unwrap().close().ok();

    let xs: Vec<f64> = rows.iter().map(|(s, g, _)| s - g).collect();
    let ys: Vec<f64> = rows.iter().map(|(_, _, w)| *w as u8 as f64).collect();
    let (b0, b1) = fit_logistic(&xs, &ys);
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
        if let Some(win) = board.win() { return (win, turn); }
        board = match board.sente() {
            true => sente.play(board),
            false => gote.play(board),
        };
    }
    unreachable!()
}

/// Newton-Raphson fit of a 2-parameter logistic regression.
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
        if det.abs() < 1e-12 { break; }
        let d0 = (h11 * g0 - h01 * g1) / det;
        let d1 = (-h01 * g0 + h00 * g1) / det;
        b0 -= d0;
        b1 -= d1;
        if d0.abs() < 1e-8 && d1.abs() < 1e-8 { break; }
    }
    (b0, b1)
}

*/
