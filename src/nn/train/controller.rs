use std::{
    fs::File,
    io::{self, Write},
    net::{Ipv4Addr, SocketAddr},
    path::PathBuf,
    sync::{Arc, Mutex},
};

use axum::{
    Json,
    body::Bytes,
    extract::{Path, Query, Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::IntoResponse,
    routing::{get, post},
};
use burn::config::Config;
use iddqd::{IdHashItem, IdHashMap, id_hash_map::RefMut, id_upcast};
use reqwest::{blocking::Client, header};
use serde::{Deserialize, Serialize};
use sha2::Digest;
use time::{UtcDateTime, format_description::well_known::Rfc3339};

use crate::nn::{
    model::{Model, ModelConfig},
    train::{
        error::{TrainError, TrainResult},
        positions::{Position, Positions, SERIALIZED_LEN},
        retry::DEFAULT_RETRY,
    },
};

const X_HEX_THIS_ITERS: header::HeaderName = header::HeaderName::from_static("x-hex-this-iters");
const X_HEX_THIS_LOSS: header::HeaderName = header::HeaderName::from_static("x-hex-this-loss");
const X_HEX_POSITIONS_CURSOR: header::HeaderName =
    header::HeaderName::from_static("x-hex-positions-cursor");

#[derive(Clone)]
pub struct ControllerClient {
    client: Client,
    controller_url: String,
}

pub enum FetchModelData {
    NotModified,
    Data(Option<String>, Vec<u8>),
}

impl FetchModelData {
    pub fn into_data(self) -> Option<(Option<String>, Vec<u8>)> {
        match self {
            FetchModelData::NotModified => None,
            FetchModelData::Data(etag, data) => Some((etag, data)),
        }
    }
}

impl ControllerClient {
    pub fn new(controller_url: String) -> ControllerClient {
        if controller_url.ends_with("/") {
            panic!("controller_url={controller_url:?} cannot end in '/'");
        }

        ControllerClient {
            client: Client::new(),
            controller_url,
        }
    }

    pub fn list_models(&self) -> TrainResult<HttpListModels> {
        DEFAULT_RETRY.attempt(
            || {
                Ok(self
                    .client
                    .get(&self.controller_url)
                    .send()?
                    .error_for_status()?
                    .json::<HttpListModels>()?)
            },
            TrainError::continue_if_retryable,
        )
    }

    pub fn create_model(&self, config: ModelConfig) -> TrainResult<()> {
        let url = format!("{}/api/model", self.controller_url);
        DEFAULT_RETRY.attempt(
            || {
                self.client
                    .post(&url)
                    .json(&config)
                    .send()?
                    .error_for_status()?;
                Ok(())
            },
            TrainError::continue_if_retryable,
        )
    }

    pub fn fetch_config(&self, model_id: &str) -> TrainResult<ModelConfig> {
        let url = format!("{}/api/model/{}/config", self.controller_url, model_id);
        DEFAULT_RETRY.attempt(
            || Ok(self.client.get(&url).send()?.error_for_status()?.json()?),
            TrainError::continue_if_retryable,
        )
    }

    pub fn fetch_model_data(
        &self,
        model_id: &str,
        etag: Option<&str>,
    ) -> TrainResult<FetchModelData> {
        let url = format!("{}/api/model/{}/params/latest", self.controller_url, model_id);
        let res = DEFAULT_RETRY.attempt(
            || {
                let mut req = self.client.get(&url);
                if let Some(etag) = etag {
                    req = req.header(header::IF_NONE_MATCH, etag);
                }
                Ok(req.send()?)
            },
            TrainError::continue_if_retryable,
        )?;
        if res.status() == StatusCode::NOT_MODIFIED {
            return Ok(FetchModelData::NotModified);
        }
        let new_etag: Option<String> = res
            .headers()
            .get(header::ETAG)
            .into_iter()
            .flat_map(|x| x.to_str().ok())
            .map(|x| x.to_owned())
            .next();
        let new_data: Vec<u8> = res.bytes()?.to_vec();
        Ok(FetchModelData::Data(new_etag, new_data))
    }

    pub fn fetch_positions(
        &self,
        model_id: &str,
        cursor: Option<isize>,
    ) -> TrainResult<(Option<usize>, Bytes)> {
        let query = match cursor {
            Some(n) => format!("?start={n}"),
            None => "".into(),
        };
        let url = format!("{}/api/model/{}/positions{}", self.controller_url, model_id, query);
        let res = DEFAULT_RETRY
            .attempt(|| Ok(self.client.get(&url).send()?), TrainError::continue_if_retryable)?;

        let cursor = res
            .headers()
            .get(X_HEX_POSITIONS_CURSOR)
            .into_iter()
            .flat_map(|x| x.to_str().into_iter())
            .flat_map(|x| x.parse::<usize>())
            .next();
        let data = res.bytes()?;

        if !data.len().is_multiple_of(SERIALIZED_LEN) {
            return Err("data length not a multiple of SERIALIZED_LEN".into());
        }

        Ok((cursor, data))
    }

    pub fn upload_positions(&self, model_id: &str, pos: Bytes) -> TrainResult<()> {
        let url = format!("{}/api/model/{}/positions", self.controller_url, model_id);
        DEFAULT_RETRY.attempt(
            || {
                self.client
                    .post(&url)
                    .body(pos.clone())
                    .send()?
                    .error_for_status()?;
                Ok(())
            },
            TrainError::continue_if_retryable,
        )
    }

    pub fn upload_params(
        &self,
        model_id: &str,
        data: Vec<u8>,
        iters: usize,
        loss: f32,
    ) -> TrainResult<()> {
        let url = format!("{}/api/model/{}/params", self.controller_url, model_id);
        let data = Bytes::from(data);
        DEFAULT_RETRY.attempt(
            || {
                self.client
                    .post(&url)
                    .header(X_HEX_THIS_ITERS, iters)
                    .header(X_HEX_THIS_LOSS, format!("{loss:.8}"))
                    .body(data.clone())
                    .send()?
                    .error_for_status()?;
                Ok(())
            },
            TrainError::continue_if_retryable,
        )
    }
}

pub struct PositionsBuffer {
    capacity: usize,
    items: Vec<Position>,
    next: usize,
    cursor: Option<usize>,
}

impl PositionsBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            items: Vec::new(),
            next: 0,
            cursor: None,
        }
    }

    pub fn poll(&mut self, client: &ControllerClient, model_id: &str) -> TrainResult<usize> {
        let cursor = match self.cursor {
            Some(n) => n as isize,
            None => -(self.capacity as isize),
        };
        let (cursor, data) = client
            .fetch_positions(model_id, Some(cursor))
            .unwrap_or_else(TrainError::unrecoverable);
        self.cursor = cursor;

        for chunk in data.chunks_exact(SERIALIZED_LEN) {
            let pos = Position::deserialize_from(chunk);
            if self.items.len() < self.capacity {
                self.items.push(pos);
            } else {
                self.items[self.next] = pos;
                self.next = (self.next + 1) % self.capacity;
            }
        }

        let n = data.len() / SERIALIZED_LEN;
        log::info!("got {n} new positions. total={}, cursor={:?}", self.items.len(), self.cursor);
        Ok(n)
    }

    pub fn count(&self) -> usize {
        self.items.len()
    }

    pub fn sample(&self, n: usize) -> impl Iterator<Item = &'_ Position> {
        let n0 = if self.items.is_empty() { 0 } else { n };
        (0..n0)
            .map(|_| rand::random_range(0..self.items.len()))
            .map(|i| &self.items[i])
    }
}

#[derive(Clone)]
struct AppConfig {
    root: PathBuf,
    port: u16,
}

impl AppConfig {
    fn load() -> Self {
        let cf = Self {
            root: std::env::var("HEX_TRAIN_ROOT")
                .map(Into::<PathBuf>::into)
                .unwrap_or("data".into()),
            port: std::env::var("PORT")
                .unwrap_or("3000".into())
                .parse::<u16>()
                .expect("PORT should parse as u16"),
        };

        if let Err(e) = std::fs::create_dir_all(&cf.root) {
            panic!("could not create {}: {e}", cf.root.display());
        }

        cf
    }
}

#[derive(Debug)]
struct ModelInfo {
    id: String,
    root: PathBuf,
    log: File,
    config: ModelConfig,
    checkpoints: Vec<String>,
    positions: Positions,
}

impl IdHashItem for ModelInfo {
    type Key<'a> = &'a str;
    fn key(&self) -> Self::Key<'_> {
        self.id.as_str()
    }
    id_upcast!();
}

impl ModelInfo {
    fn init(config: ModelConfig, root: PathBuf) -> io::Result<ModelInfo> {
        let id = config.id();
        log::info!("initializing {id}");

        std::fs::create_dir_all(&root)?;
        config.save(root.join("config.json"))?;

        let log = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(root.join("train-log.txt"))?;

        let checkpoints = Vec::new();
        let positions = Positions::open(&root.join("positions"))?;
        let mut info = ModelInfo {
            id,
            root,
            log,
            config,
            checkpoints,
            positions,
        };

        let model: Model<burn::backend::NdArray> = info.config.init(&Default::default());
        let bytes = model.into_bytes();
        info.write_checkpoint(&bytes, None, None)?;

        Ok(info)
    }

    fn load(id: String, root: PathBuf) -> io::Result<ModelInfo> {
        log::info!("loading {id}");

        let mut checkpoints: Vec<String> = Vec::new();
        for item in std::fs::read_dir(&root)? {
            let item = item?;
            let path = item.path();
            let file_name = item
                .file_name()
                .into_string()
                .expect("model dir entry should be a valid UTF-8 string");
            if !item.file_type()?.is_file() {
                log::warn!("skipping {}: not a file", path.display());
            } else if file_name.starts_with("checkpoint-") {
                checkpoints.push(file_name);
            } else if file_name == "config.json" {
                // ignore silently
            } else if file_name == "positions" {
                // ignore silently
            } else if file_name == "train-log.txt" {
                // ignore silently
            } else {
                log::warn!("skipping {}: not sure what this is", path.display());
            }
        }
        checkpoints.sort();

        let log = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(root.join("train-log.txt"))?;

        let config = ModelConfig::load(&root.join("config.json")).map_err(io::Error::other)?;
        let positions: Positions = Positions::open(&root.join("positions"))?;

        Ok(ModelInfo {
            id,
            root,
            log,
            config,
            checkpoints,
            positions,
        })
    }

    fn path_to_latest(&self) -> Option<PathBuf> {
        self.checkpoints.last().map(|x| self.root.join(x))
    }

    fn write_checkpoint(
        &mut self,
        data: &[u8],
        this_iters: Option<usize>,
        this_loss: Option<f32>,
    ) -> io::Result<String> {
        let hash: u32 = {
            let bytes: [u8; 32] = sha2::Sha256::digest(data).try_into().unwrap();
            u32::from_be_bytes(bytes[..4].try_into().unwrap())
        };
        let name = format!("checkpoint-{:05}-{:08x}", self.checkpoints.len(), hash);

        self.checkpoints.push(name.clone());
        std::fs::write(self.root.join(&name), data)?;
        log::info!("{}: wrote {}", self.id, name);

        let log_time = UtcDateTime::now().format(&Rfc3339).unwrap();
        let log_iters = match this_iters {
            Some(it) => format!("{it}"),
            None => String::new(),
        };
        let log_loss = match this_loss {
            Some(it) => format!("{it:.8}"),
            None => String::new(),
        };
        write!(self.log, "{log_time},{name},{log_iters},{log_loss},{}\n", self.positions.len())
            .inspect_err(|e| log::warn!("failed to write to log: {e}"))
            .ok();
        self.log
            .flush()
            .inspect_err(|e| log::warn!("failed to flush log: {e}"))
            .ok();

        Ok(name)
    }
}

#[derive(Clone)]
struct AppState {
    cf: Arc<AppConfig>,
    models: Arc<Mutex<IdHashMap<ModelInfo>>>,
}

impl AppState {
    fn new(cf: AppConfig) -> io::Result<Self> {
        let models_dir = cf.root.join("models");

        if let Err(e) = std::fs::create_dir_all(&models_dir) {
            panic!("could not create {}: {e}", models_dir.display());
        }

        let mut models = IdHashMap::<ModelInfo>::new();
        for item in std::fs::read_dir(&models_dir).unwrap() {
            let item = item.unwrap();
            if !item.file_type().unwrap().is_dir() {
                log::warn!("skipping {}: not a directory", item.path().display());
            } else {
                let file_name = item
                    .file_name()
                    .into_string()
                    .expect("model dir name should be a valid UTF-8 string");
                models
                    .insert_unique(ModelInfo::load(file_name, item.path())?)
                    .expect("model IDs should be unique");
            }
        }

        Ok(Self {
            cf: Arc::new(cf),
            models: Arc::new(Mutex::new(models)),
        })
    }

    fn create_model(&self, config: ModelConfig) -> io::Result<String> {
        let mut models = self.models.lock().unwrap();
        let id = config.id();
        if !models.contains_key(&id.as_str()) {
            let path = self.cf.root.join("models").join(&id);
            models
                .insert_unique(ModelInfo::init(config, path)?)
                .map_err(|_| io::Error::other("duplicate id"))?;
        }
        Ok(id)
    }

    fn with_model<F, T>(&self, id: &str, cb: F) -> Option<T>
    where
        F: FnOnce(&ModelInfo) -> T,
    {
        self.models.lock().unwrap().get(id).map(cb)
    }

    fn with_model_mut<F, T>(&self, id: &str, cb: F) -> Option<T>
    where
        F: FnOnce(RefMut<'_, ModelInfo>) -> T,
    {
        self.models.lock().unwrap().get_mut(id).map(cb)
    }
}

#[derive(Serialize, Deserialize)]
pub struct HttpListModels {
    pub models: Vec<HttpListModelsItem>,
}
#[derive(Serialize, Deserialize)]
pub struct HttpListModelsItem {
    pub id: String,
    pub config: ModelConfig,
    pub checkpoints: Vec<String>,
}
async fn http_get_root(State(app): State<AppState>) -> impl IntoResponse {
    let models: Vec<_> = app
        .models
        .lock()
        .unwrap()
        .iter()
        .map(|x| HttpListModelsItem {
            id: x.id.clone(),
            config: x.config.clone(),
            checkpoints: x.checkpoints.clone(),
        })
        .collect();
    Json(HttpListModels { models })
}

async fn http_api_post_model(
    State(app): State<AppState>,
    Json(config): Json<ModelConfig>,
) -> impl IntoResponse {
    app.create_model(config).map_err(|e| {
        log::error!("{e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })
}

async fn http_api_get_model_config(
    State(app): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let Some(res) = app.with_model(&id, |m| m.config.clone()) else {
        return Err(StatusCode::NOT_FOUND);
    };
    Ok(Json(res))
}

async fn http_api_post_model_params(
    State(app): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let iters = headers
        .get(X_HEX_THIS_ITERS)
        .and_then(|x| x.to_str().ok())
        .and_then(|x| x.parse::<usize>().ok());
    let loss = headers
        .get(X_HEX_THIS_LOSS)
        .and_then(|x| x.to_str().ok())
        .and_then(|x| x.parse::<f32>().ok());
    let Some(res) = app.with_model_mut(&id, |mut m| m.write_checkpoint(&body, iters, loss)) else {
        return Err(StatusCode::NOT_FOUND);
    };
    let Ok(_) = res else {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    };
    Ok(())
}

async fn http_api_get_model_params_latest(
    State(app): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let Some(path) = app.with_model(&id, |m| m.path_to_latest()).flatten() else {
        return Err(StatusCode::NOT_FOUND);
    };

    let name = path
        .file_name()
        .expect("model path should have filename")
        .to_str()
        .expect("model file name should be valid UTF-8");
    let etag = format!("\"{name}\"");

    if let Some(x) = headers.get(header::IF_NONE_MATCH)
        && x.as_bytes() == etag.as_bytes()
    {
        return Err(StatusCode::NOT_MODIFIED);
    }

    let Ok(data) = std::fs::read(&path) else {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    };
    Ok(([(header::ETAG, etag)], data))
}

async fn http_api_post_positions(
    State(app): State<AppState>,
    Path(id): Path<String>,
    body: Bytes,
) -> impl IntoResponse {
    if !body.len().is_multiple_of(SERIALIZED_LEN) {
        return Err(StatusCode::BAD_REQUEST);
    }
    let Some(res) = app.with_model_mut(&id, |mut m| m.positions.push_serialized_many(&body)) else {
        return Err(StatusCode::NOT_FOUND);
    };
    let Ok(_) = res else {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    };
    Ok(())
}

#[derive(Serialize, Deserialize, Debug)]
struct HttpGetPositionsQuery {
    start: Option<isize>,
}
async fn http_api_get_positions(
    State(app): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<HttpGetPositionsQuery>,
) -> impl IntoResponse {
    let Some(res) = app.with_model_mut(&id, |mut m| {
        let n = m.positions.len() as isize;
        let (start, end) = match query.start {
            Some(i) if i < 0 => ((n + i).max(0) as usize, None),
            Some(i) => (i as usize, None),
            None => (0, None),
        };
        m.positions.read_serialized_range(start, end)
    }) else {
        return Err(StatusCode::NOT_FOUND);
    };
    let Ok((data, end)) = res else {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    };
    Ok(([(X_HEX_POSITIONS_CURSOR, format!("{end}"))], data))
}

#[tokio::main]
pub async fn main() {
    let cf = AppConfig::load();
    let app = axum::Router::new()
        .route("/", get(http_get_root))
        .route("/api/model", post(http_api_post_model))
        .route("/api/model/{id}/config", get(http_api_get_model_config))
        .route("/api/model/{id}/params", post(http_api_post_model_params))
        .route("/api/model/{id}/params/latest", get(http_api_get_model_params_latest))
        .route("/api/model/{id}/positions", post(http_api_post_positions))
        .route("/api/model/{id}/positions", get(http_api_get_positions))
        .with_state(AppState::new(cf.clone()).unwrap())
        .layer(axum::middleware::from_fn(async |req: Request, next: Next| {
            let msg = format!("{} {}", req.method(), req.uri());
            let res = next.run(req).await;
            log::info!("{msg} - {}", res.status());
            res
        }));

    let addr: SocketAddr = (Ipv4Addr::UNSPECIFIED, cf.port).into();
    log::info!("listening on {addr:?}");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("could not bind listener");
    axum::serve(listener, app)
        .await
        .expect("http listener exited");
}
