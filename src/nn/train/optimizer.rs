use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc::{Receiver, Sender, channel},
    },
    time::{Duration, Instant},
};

use burn::optim::{
    GradientsParams, Optimizer, SgdConfig, decay::WeightDecayConfig, momentum::MomentumConfig,
};

use crate::nn::{
    model::{Model, positions_to_input},
    train::{
        controller::{ControllerClient, PositionsBuffer},
        error::TrainError,
        metrics::CounterLog,
    },
};

type Wgpu = burn::backend::Autodiff<burn::backend::Wgpu<f32, i32>>;

#[derive(Clone)]
struct AppConfig {
    client: ControllerClient,
    to_uploader: Sender<UploaderMsg>,
    model_id: String,
    upload_interval_secs: u64,
    batch_size: usize,
    momentum: f64,
    learning_rate: f64,
    positions: Arc<Mutex<PositionsBuffer>>,
    losses: Arc<Mutex<Vec<f32>>>,
    total_iters: Arc<AtomicUsize>,
}

impl AppConfig {
    fn load(to_uploader: Sender<UploaderMsg>) -> Self {
        let controller_url = std::env::var("HEX_TRAIN_CONTROLLER_URL")
            .expect("HEX_TRAIN_CONTROLLER_URL is a required env var");
        let max_positions = std::env::var("HEX_TRAIN_OPTIMIZER_MAX_POSITIONS")
            .unwrap_or("500000".into())
            .parse::<usize>()
            .expect("HEX_TRAIN_OPTIMIZER_MAX_POSITIONS should parse as usize");
        log::info!("HEX_TRAIN_CONTROLLER_URL={}", controller_url);
        log::info!("HEX_TRAIN_OPTIMIZER_MAX_POSITIONS={}", max_positions);

        let cf = Self {
            client: ControllerClient::new(controller_url),
            to_uploader,
            model_id: std::env::var("HEX_TRAIN_MODEL_ID")
                .expect("HEX_TRAIN_MODEL_ID is a required env var"),
            upload_interval_secs: std::env::var("HEX_TRAIN_OPTIMIZER_UPLOAD_INTERVAL")
                .unwrap_or("300".into())
                .parse::<u64>()
                .expect("HEX_TRAIN_OPTIMIZER_UPLOAD_INTERVAL should parse as u64"),
            batch_size: std::env::var("HEX_TRAIN_OPTIMIZER_BATCH_SIZE")
                .unwrap_or("256".into())
                .parse::<usize>()
                .expect("HEX_TRAIN_OPTIMIZER_BATCH_SIZE should parse as usize"),
            momentum: std::env::var("HEX_TRAIN_OPTIMIZER_MOMENTUM")
                .unwrap_or("0.7".into())
                .parse::<f64>()
                .expect("HEX_TRAIN_OPTIMIZER_MOMENTUM should parse as f64"),
            learning_rate: std::env::var("HEX_TRAIN_OPTIMIZER_LEARNING_RATE")
                .unwrap_or("0.02".into())
                .parse::<f64>()
                .expect("HEX_TRAIN_OPTIMIZER_LEARNING_RATE should parse as f64"),
            positions: Arc::new(Mutex::new(PositionsBuffer::new(max_positions))),
            losses: Arc::new(Mutex::new(Vec::new())),
            total_iters: Arc::new(AtomicUsize::new(0)),
        };
        log::info!("HEX_TRAIN_MODEL_ID={}", cf.model_id);
        log::info!("HEX_TRAIN_OPTIMIZER_UPLOAD_INTERVAL={}", cf.upload_interval_secs);
        log::info!("HEX_TRAIN_OPTIMIZER_BATCH_SIZE={}", cf.batch_size);
        log::info!("HEX_TRAIN_OPTIMIZER_MOMENTUM={}", cf.momentum);
        log::info!("HEX_TRAIN_OPTIMIZER_LEARNING_RATE={}", cf.learning_rate);

        cf
    }
}

fn spawn_optimizer(cf: AppConfig) {
    std::thread::spawn(move || optimizer(cf));
}

fn optimizer(cf: AppConfig) {
    let device = Default::default();
    let config = cf.client.fetch_config(&cf.model_id).unwrap();
    let (_, data) = cf
        .client
        .fetch_model_data(&cf.model_id, None)
        .unwrap_or_else(TrainError::unrecoverable)
        .into_data()
        .expect("fetch without etag should always return data");
    let mut model: Model<Wgpu> = config.init(&device).load_bytes(data, &device);
    let mut optim = SgdConfig::new()
        .with_momentum(Some(MomentumConfig::new().with_momentum(cf.momentum)))
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .init::<Wgpu, Model<Wgpu>>();

    let mut last_upload = Instant::now();

    for _ in 0.. {
        let positions = {
            let buf = cf.positions.lock().unwrap();
            if buf.count() < cf.batch_size {
                log::info!("waiting a bit for position buffer to fill");
                std::mem::drop(buf);
                std::thread::sleep(Duration::from_secs(60));
                continue;
            }
            positions_to_input(buf.sample(cf.batch_size), &device)
        };

        let loss = model.forward_loss(positions);
        cf.total_iters.fetch_add(1, Ordering::Relaxed);
        cf.losses.lock().unwrap().push(loss.clone().into_scalar());

        let grad = loss.backward();
        let grad = GradientsParams::from_grads(grad, &model);
        model = optim.step(cf.learning_rate, model, grad);

        if last_upload.elapsed() >= Duration::from_secs(cf.upload_interval_secs) {
            last_upload = Instant::now();
            log::info!("uploading new model checkpoint");
            cf.to_uploader
                .send(UploaderMsg::Queue(model.clone().into_bytes()))
                .unwrap();
        }
    }
}

fn spawn_poller(cf: AppConfig) {
    std::thread::spawn(move || positions_poller(cf));
}

fn positions_poller(cf: AppConfig) {
    loop {
        cf.positions
            .lock()
            .unwrap()
            .poll(&cf.client, &cf.model_id)
            .unwrap_or_else(TrainError::unrecoverable);
        std::thread::sleep(Duration::from_secs(60));
    }
}

enum UploaderMsg {
    Queue(Vec<u8>),
}

fn spawn_uploader(cf: AppConfig, inbox: Receiver<UploaderMsg>) {
    std::thread::spawn(move || uploader(cf, inbox));
}

fn uploader(cf: AppConfig, inbox: Receiver<UploaderMsg>) {
    loop {
        let Ok(UploaderMsg::Queue(data)) = inbox.recv() else {
            panic!();
        };
        if let Err(e) = cf.client.upload_params(&cf.model_id, data) {
            log::error!("failed to upload params: {e}");
        }
    }
}

pub fn main() {
    let (uploader_send, uploader_recv) = channel();
    let cf = AppConfig::load(uploader_send);

    spawn_optimizer(cf.clone());
    spawn_poller(cf.clone());
    spawn_uploader(cf.clone(), uploader_recv);

    let mut log_iters = CounterLog::new();
    loop {
        std::thread::sleep(Duration::from_secs(60));
        log_iters.report(&cf.total_iters);
        let losses = std::mem::take(&mut *cf.losses.lock().unwrap());
        let loss_count = losses.len() as f32;
        let loss = losses.into_iter().sum::<f32>() / loss_count;
        log::info!(
            "iters={:07} ({:.2}/s) loss={:.8}",
            log_iters.latest(),
            log_iters.per_second(),
            loss
        );
    }
}
