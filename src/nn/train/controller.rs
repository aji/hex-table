use std::{
    io,
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
use reqwest::header;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::Digest;

use crate::nn::{
    model::{Model, ModelConfig},
    train::positions::{Positions, SERIALIZED_LEN},
};

const X_HEX_POSITIONS_CURSOR: header::HeaderName =
    header::HeaderName::from_static("x-hex-positions-cursor");

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

        let checkpoints = Vec::new();
        let positions = Positions::open(&root.join("positions"))?;
        let mut info = ModelInfo {
            id,
            root,
            config,
            checkpoints,
            positions,
        };

        let model: Model<burn::backend::Cpu> = info.config.init(&Default::default());
        let bytes = model.into_bytes();
        info.write_checkpoint(&bytes)?;

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
            } else {
                log::warn!("skipping {}: not sure what this is", path.display());
            }
        }
        checkpoints.sort();

        let config = ModelConfig::load(&root.join("config.json")).map_err(io::Error::other)?;
        let positions: Positions = Positions::open(&root.join("positions"))?;

        Ok(ModelInfo {
            id,
            root,
            config,
            checkpoints,
            positions,
        })
    }

    fn path_to_latest(&self) -> Option<PathBuf> {
        self.checkpoints.last().map(|x| self.root.join(x))
    }

    fn write_checkpoint(&mut self, data: &[u8]) -> io::Result<String> {
        let hash: u32 = {
            let bytes: [u8; 32] = sha2::Sha256::digest(data).try_into().unwrap();
            u32::from_be_bytes(bytes[..4].try_into().unwrap())
        };
        let name = format!("checkpoint-{:05}-{:08x}", self.checkpoints.len(), hash);
        self.checkpoints.push(name.clone());
        std::fs::write(self.root.join(&name), data)?;
        log::info!("{}: wrote {}", self.id, name);
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

async fn http_get_root(State(app): State<AppState>) -> impl IntoResponse {
    Json(json!({
        "root": app.cf.root.display().to_string(),
    }))
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
    body: Bytes,
) -> impl IntoResponse {
    let Some(res) = app.with_model_mut(&id, |mut m| m.write_checkpoint(&body)) else {
        return Err(StatusCode::NOT_FOUND);
    };
    let Ok(_) = res else {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    };
    Ok(())
}

async fn http_api_get_model_params_latest(
    headers: HeaderMap,
    State(app): State<AppState>,
    Path(id): Path<String>,
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
