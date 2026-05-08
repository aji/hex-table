use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc::{Receiver, RecvTimeoutError, Sender, SyncSender, channel, sync_channel},
    },
    time::Duration,
};

use bytes::BytesMut;

use crate::{
    bb::Bitboard,
    nn::{
        model::{EvalRequest, EvalResult, Model, ModelConfig},
        search::{Evaluator, search_with_evaluator},
        train::{
            controller::{ControllerClient, FetchModelData},
            error::TrainError,
            metrics::CounterLog,
            positions::{Position, SERIALIZED_LEN},
        },
        transform::Transpose,
    },
};

type Wgpu = burn::backend::Wgpu<f32, i32>;

#[derive(Clone)]
struct AppConfig {
    client: ControllerClient,
    to_evaluator: Sender<EvaluatorMsg>,
    to_uploader: Sender<UploaderMsg>,
    model_id: String,
    batch_evals: usize,
    concurrency: usize,
    iters: usize,
    total_iters: Arc<AtomicUsize>,
    total_moves: Arc<AtomicUsize>,
    total_games: Arc<AtomicUsize>,
    total_games_sente: Arc<AtomicUsize>,
}

impl AppConfig {
    fn load(eval: Sender<EvaluatorMsg>, pos: Sender<UploaderMsg>) -> Self {
        let controller_url = std::env::var("HEX_TRAIN_CONTROLLER_URL")
            .expect("HEX_TRAIN_CONTROLLER_URL is a required env var");
        log::info!("HEX_TRAIN_CONTROLLER_URL={}", controller_url);

        let cf = Self {
            client: ControllerClient::new(controller_url),
            to_evaluator: eval,
            to_uploader: pos,
            model_id: std::env::var("HEX_TRAIN_MODEL_ID")
                .expect("HEX_TRAIN_MODEL_ID is a required env var"),
            batch_evals: std::env::var("HEX_TRAIN_SELF_PLAY_BATCH_EVALS")
                .unwrap_or("32".into())
                .parse::<usize>()
                .expect("HEX_TRAIN_SELF_PLAY_BATCH_EVALS should parse as usize"),
            concurrency: std::env::var("HEX_TRAIN_SELF_PLAY_CONCURRENCY")
                .unwrap_or("128".into())
                .parse::<usize>()
                .expect("HEX_TRAIN_SELF_PLAY_CONCURRENCY should parse as usize"),
            iters: std::env::var("HEX_TRAIN_SELF_PLAY_ITERS")
                .unwrap_or("800".into())
                .parse::<usize>()
                .expect("HEX_TRAIN_SELF_PLAY_ITERS should parse as usize"),
            total_iters: Arc::new(AtomicUsize::new(0)),
            total_moves: Arc::new(AtomicUsize::new(0)),
            total_games: Arc::new(AtomicUsize::new(0)),
            total_games_sente: Arc::new(AtomicUsize::new(0)),
        };

        if cf.concurrency <= cf.batch_evals {
            panic!(
                "concurrency={} must be greater than batch_evals={}",
                cf.concurrency, cf.batch_evals
            );
        }

        log::info!("HEX_TRAIN_MODEL_ID={}", cf.model_id);
        log::info!("HEX_TRAIN_SELF_PLAY_BATCH_EVALS={}", cf.batch_evals);
        log::info!("HEX_TRAIN_SELF_PLAY_CONCURRENCY={}", cf.concurrency);
        log::info!("HEX_TRAIN_SELF_PLAY_ITERS={}", cf.iters);

        cf
    }
}

impl Evaluator for AppConfig {
    fn call(&self, board: Bitboard) -> EvalResult {
        let (send, recv) = sync_channel(1);
        self.to_evaluator
            .send(EvaluatorMsg::Queue(board, send))
            .unwrap();
        recv.recv().unwrap()
    }
}

type EvaluatorRet = SyncSender<EvalResult>;

enum EvaluatorMsg {
    Init(ModelConfig, Vec<u8>),
    ModelUpdated(Vec<u8>),
    Queue(Bitboard, EvaluatorRet),
}

fn spawn_evaluator(cf: AppConfig, inbox: Receiver<EvaluatorMsg>) {
    std::thread::spawn(move || evaluator(cf, inbox));
}

fn evaluator(cf: AppConfig, inbox: Receiver<EvaluatorMsg>) {
    let device = Default::default();
    let mut model: Model<Wgpu> = match inbox.recv() {
        Ok(EvaluatorMsg::Init(config, data)) => config.init(&device).load_bytes(data, &device),
        _ => panic!("expected an init message"),
    };

    log::info!("evaluator ready");

    let mut reqs: Vec<EvalRequest> = Vec::new();
    let mut rets: Vec<EvaluatorRet> = Vec::new();

    loop {
        let go = match inbox.recv() {
            Ok(EvaluatorMsg::Init(_, _)) => {
                panic!("received unexpected init message");
            }
            Ok(EvaluatorMsg::ModelUpdated(data)) => {
                log::info!("loading updated model");
                model = model.load_bytes(data, &device);
                false
            }
            Ok(EvaluatorMsg::Queue(board, ret)) => {
                reqs.push(EvalRequest::new(board));
                rets.push(ret);
                reqs.len() >= cf.batch_evals
            }
            Err(e) => panic!("recv error: {e}"),
        };

        if !go {
            continue;
        }

        assert_eq!(reqs.len(), rets.len());
        let reqs = std::mem::take(&mut reqs);
        let rets = std::mem::take(&mut rets);
        for (ret, res) in rets
            .into_iter()
            .zip(model.eval_batch(reqs, &device).into_iter())
        {
            ret.send(res).unwrap();
        }
    }
}

enum UploaderMsg {
    Queue(Vec<(Bitboard, [f32; 121])>, Bitboard),
}

fn spawn_uploader(cf: AppConfig, inbox: Receiver<UploaderMsg>) {
    std::thread::spawn(move || uploader(cf, inbox));
}

fn uploader(cf: AppConfig, inbox: Receiver<UploaderMsg>) {
    let mut pending: BytesMut = Default::default();

    loop {
        let go = match inbox.recv_timeout(Duration::from_secs(30)) {
            Ok(UploaderMsg::Queue(log, board)) => {
                let value = match board.win() {
                    Some(true) => 1.0,
                    Some(false) => -1.0,
                    None => {
                        log::error!("non-final state sent to uploader");
                        continue;
                    }
                };
                for (board, policy) in log.into_iter() {
                    let mut pos = Position {
                        board,
                        value,
                        policy: policy.try_into().unwrap(),
                    };
                    if !pos.board.sente() {
                        pos.apply_transform(&Transpose);
                    }
                    pos.serialize_into(&mut pending);
                }
                pending.len() > 1_000_000
            }
            Err(RecvTimeoutError::Disconnected) => {
                panic!("uploader disconnected");
            }
            Err(RecvTimeoutError::Timeout) => pending.len() > 0,
        };

        if !go {
            continue;
        }

        log::info!("flushing uploader buffer ({} positions)", pending.len() / SERIALIZED_LEN);
        let positions = std::mem::take(&mut pending).freeze();
        cf.client
            .upload_positions(&cf.model_id, positions)
            .unwrap_or_else(TrainError::unrecoverable);
    }
}

fn spawn_fetcher(cf: AppConfig) {
    let config = cf
        .client
        .fetch_config(&cf.model_id)
        .unwrap_or_else(TrainError::unrecoverable);
    let (etag, data) = cf
        .client
        .fetch_model_data(&cf.model_id, None)
        .unwrap_or_else(TrainError::unrecoverable)
        .into_data()
        .expect("fetch without etag should always return data");
    cf.to_evaluator
        .send(EvaluatorMsg::Init(config, data))
        .unwrap();
    std::thread::spawn(move || fetcher(cf, etag));
}

fn fetcher(cf: AppConfig, mut etag: Option<String>) {
    loop {
        std::thread::sleep(Duration::from_secs(30));
        let res = cf
            .client
            .fetch_model_data(&cf.model_id, etag.as_deref())
            .unwrap_or_else(TrainError::unrecoverable);
        if let FetchModelData::Data(new_etag, data) = res {
            etag = new_etag;
            cf.to_evaluator
                .send(EvaluatorMsg::ModelUpdated(data))
                .unwrap();
        }
    }
}

fn spawn_self_play(cf: AppConfig) {
    for idx in 0..cf.concurrency {
        let cf = cf.clone();
        std::thread::spawn(move || self_play(idx, cf));
    }
}

fn self_play(_idx: usize, cf: AppConfig) {
    let monitor = |it: usize| {
        cf.total_iters.fetch_add(1, Ordering::Relaxed);
        it < cf.iters
    };
    loop {
        let mut board = Bitboard::new();
        let mut log: Vec<(Bitboard, [f32; 121])> = Vec::new();
        while board.win().is_none() {
            let out = search_with_evaluator(&cf, board, 0.25, monitor);
            cf.total_moves.fetch_add(1, Ordering::Relaxed);
            log.push((board, out.policy.try_into().unwrap()));
            board = if log.len() < 30 { out.board_sample } else { out.board_best };
        }
        cf.total_games.fetch_add(1, Ordering::Relaxed);
        if let Some(true) = board.win() {
            cf.total_games_sente.fetch_add(1, Ordering::Relaxed);
        }
        cf.to_uploader.send(UploaderMsg::Queue(log, board)).unwrap();
    }
}

pub fn main() {
    let (eval_send, eval_recv) = channel();
    let (pos_send, pos_recv) = channel();

    let cf = AppConfig::load(eval_send, pos_send);

    spawn_evaluator(cf.clone(), eval_recv);
    spawn_uploader(cf.clone(), pos_recv);
    spawn_fetcher(cf.clone());
    spawn_self_play(cf.clone());

    let mut log_iters = CounterLog::new();
    let mut log_moves = CounterLog::new();
    let mut log_games = CounterLog::new();
    let mut log_games_sente = CounterLog::new();
    loop {
        std::thread::sleep(Duration::from_secs(10));

        log_iters.report(&cf.total_iters);
        log_moves.report(&cf.total_moves);
        log_games.report(&cf.total_games);
        log_games_sente.report(&cf.total_games_sente);

        log::info!(
            "iters={} ({:.1}/s), moves={} ({:.2}s/), games={} ({:.1}m/), sente={:.1}% (recent {:.1}%)",
            log_iters.latest(),
            log_iters.per_second(),
            log_moves.latest(),
            log_moves.seconds_per(),
            log_games.latest(),
            log_games.seconds_per() / 60.0,
            log_games_sente.latest() as f64 * 100.0 / log_games.latest() as f64,
            log_games_sente.count() as f64 * 100.0 / log_games.count() as f64,
        );
    }
}
