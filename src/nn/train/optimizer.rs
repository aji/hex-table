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
use reqwest::{blocking::Client, header};

use crate::nn::{
    model::{Model, ModelConfig, positions_to_input},
    train::{
        metrics::CounterLog,
        positions::{Position, SERIALIZED_LEN},
    },
};

type Wgpu = burn::backend::Autodiff<burn::backend::Wgpu<f32, i32>>;

const X_HEX_POSITIONS_CURSOR: header::HeaderName =
    header::HeaderName::from_static("x-hex-positions-cursor");

#[derive(Clone)]
struct AppConfig {
    client: Client,
    to_uploader: Sender<UploaderMsg>,
    controller_url: String,
    model_id: String,
    upload_interval_secs: u64,
    max_positions: usize,
    batch_size: usize,
    momentum: f64,
    positions: Arc<Mutex<PositionsBuffer>>,
    total_iters: Arc<AtomicUsize>,
}

impl AppConfig {
    fn load(to_uploader: Sender<UploaderMsg>) -> Self {
        let cf = Self {
            client: Client::new(),
            to_uploader,
            controller_url: std::env::var("HEX_TRAIN_CONTROLLER_URL")
                .expect("HEX_TRAIN_CONTROLLER_URL is a required env var"),
            model_id: std::env::var("HEX_TRAIN_MODEL_ID")
                .expect("HEX_TRAIN_MODEL_ID is a required env var"),
            upload_interval_secs: std::env::var("HEX_TRAIN_UPLOAD_INTERVAL")
                .unwrap_or("300".into())
                .parse::<u64>()
                .expect("HEX_TRAIN_UPLOAD_INTERVAL should parse as u64"),
            max_positions: std::env::var("HEX_TRAIN_MAX_POSITIONS")
                .unwrap_or("500000".into())
                .parse::<usize>()
                .expect("HEX_TRAIN_MAX_POSITIONS should parse as usize"),
            batch_size: std::env::var("HEX_TRAIN_BATCH_SIZE")
                .unwrap_or("256".into())
                .parse::<usize>()
                .expect("HEX_TRAIN_BATCH_SIZE should parse as usize"),
            momentum: std::env::var("HEX_TRAIN_MOMENTUM")
                .unwrap_or("0.7".into())
                .parse::<f64>()
                .expect("HEX_TRAIN_MOMENTUM should parse as f64"),
            positions: Arc::new(Mutex::new(PositionsBuffer::new())),
            total_iters: Arc::new(AtomicUsize::new(0)),
        };

        if cf.controller_url.ends_with("/") {
            panic!("controller_url={:?} cannot end in '/'", cf.controller_url);
        }

        log::info!("HEX_TRAIN_CONTROLLER_URL={}", cf.controller_url);
        log::info!("HEX_TRAIN_MODEL_ID={}", cf.model_id);
        log::info!("HEX_TRAIN_UPLOAD_INTERVAL={}", cf.upload_interval_secs);
        log::info!("HEX_TRAIN_MAX_POSITIONS={}", cf.max_positions);
        log::info!("HEX_TRAIN_BATCH_SIZE={}", cf.batch_size);
        log::info!("HEX_TRAIN_MOMENTUM={}", cf.momentum);

        cf
    }

    fn fetch_config(&self) -> reqwest::Result<ModelConfig> {
        let url = format!("{}/api/model/{}/config", self.controller_url, self.model_id);
        self.client.get(&url).send()?.json()
    }

    fn fetch_model_data(&self) -> reqwest::Result<Vec<u8>> {
        let url = format!("{}/api/model/{}/params/latest", self.controller_url, self.model_id);
        Ok(self.client.get(&url).send()?.bytes()?.to_vec())
    }

    fn upload_params(&self, data: Vec<u8>) -> reqwest::Result<()> {
        let url = format!("{}/api/model/{}/params", self.controller_url, self.model_id);
        self.client
            .post(&url)
            .body(data)
            .send()?
            .error_for_status()?;
        Ok(())
    }
}

struct PositionsBuffer {
    items: Vec<Position>,
    next: usize,
    cursor: Option<usize>,
}

impl PositionsBuffer {
    fn new() -> Self {
        Self {
            items: Vec::new(),
            next: 0,
            cursor: None,
        }
    }

    fn poll(&mut self, cf: &AppConfig) -> Option<()> {
        let query = match self.cursor {
            Some(n) => format!("start={n}"),
            None => format!("start=-{}", cf.max_positions),
        };
        let url = format!("{}/api/model/{}/positions?{query}", cf.controller_url, cf.model_id);
        let res = cf.client.get(&url).send().ok()?;

        let cursor = res
            .headers()
            .get(X_HEX_POSITIONS_CURSOR)
            .into_iter()
            .flat_map(|x| x.to_str().into_iter())
            .flat_map(|x| x.parse::<usize>())
            .next();

        let data = res.bytes().ok()?.to_vec();
        if !data.len().is_multiple_of(SERIALIZED_LEN) {
            log::error!("data length is not a multiple of SERIALIZED_LEN");
            return None;
        }
        if data.len() == 0 {
            return Some(());
        }

        let n = data.len() / SERIALIZED_LEN;
        for i in 0..n {
            let i0 = i * SERIALIZED_LEN;
            let i1 = (i + 1) * SERIALIZED_LEN;
            let pos = Position::deserialize_from(&data[i0..i1]);
            if self.items.len() < cf.max_positions {
                self.items.push(pos);
            } else {
                self.items[self.next] = pos;
                self.next = (self.next + 1) % cf.max_positions;
            }
        }
        self.cursor = cursor;

        log::info!("got {n} new positions. total={}, cursor={:?}", self.items.len(), self.cursor);
        Some(())
    }

    fn count(&self) -> usize {
        self.items.len()
    }

    fn sample(&self, n: usize) -> impl Iterator<Item = &'_ Position> {
        let n0 = if self.items.is_empty() { 0 } else { n };
        (0..n0)
            .map(|_| rand::random_range(0..self.items.len()))
            .map(|i| &self.items[i])
    }
}

fn spawn_optimizer(cf: AppConfig) {
    std::thread::spawn(move || optimizer(cf));
}

fn optimizer(cf: AppConfig) {
    let device = Default::default();
    let config = cf.fetch_config().unwrap();
    let data = cf.fetch_model_data().unwrap();
    let mut model: Model<Wgpu> = config.init(&device).load_bytes(data, &device);
    let mut optim = SgdConfig::new()
        .with_momentum(Some(MomentumConfig::new().with_momentum(cf.momentum)))
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .init::<Wgpu, Model<Wgpu>>();
    let lr = |_| 1e-2;

    let mut last_print = Instant::now();
    let mut last_upload = Instant::now();
    let mut loss_total = 0.0;
    let mut loss_count = 0;

    for iter in 0.. {
        let positions = {
            let buf = cf.positions.lock().unwrap();
            if buf.count() < cf.batch_size {
                log::info!("waiting a bit for position buffer to fill");
                std::mem::drop(buf);
                std::thread::sleep(Duration::from_secs(10));
                continue;
            }
            positions_to_input(buf.sample(cf.batch_size), &device)
        };

        let loss = model.forward_loss(positions);
        cf.total_iters.fetch_add(1, Ordering::Relaxed);
        loss_total += loss.clone().into_scalar();
        loss_count += 1;
        if last_print.elapsed() >= Duration::from_secs(1) {
            log::info!("loss={:10.8}", loss_total / loss_count as f32);
            last_print = Instant::now();
            loss_total = 0.0;
            loss_count = 0;
        }

        let grad = loss.backward();
        let grad = GradientsParams::from_grads(grad, &model);
        model = optim.step(lr(iter), model, grad);

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
        cf.positions.lock().unwrap().poll(&cf);
        std::thread::sleep(Duration::from_secs(10));
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
        if let Err(e) = cf.upload_params(data) {
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
        std::thread::sleep(Duration::from_secs(10));
        log_iters.report(&cf.total_iters);
        log::info!("iters={} ({:.1}/s)", log_iters.latest(), log_iters.per_second());
    }
}
