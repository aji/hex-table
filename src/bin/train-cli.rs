use std::{io::Write, time::Instant};

use clap::{Parser, Subcommand};
use hex_table::{
    bb::{Bitboard, BitboardPretty},
    nn::{
        burn::train::{
            controller::ControllerClient,
            error::TrainError,
            positions::{Position, SERIALIZED_LEN},
        },
        model::{EvalRequest, Model, ModelConfig},
        search::search,
        transform::Transpose,
    },
    util::{Finite, IteratorExt},
};
use rand::SeedableRng;

#[derive(Parser, Debug)]
struct Cli {
    /// The controller URL. Defaults to HEX_TRAIN_CONTROLLER_URL
    #[arg(long, value_name = "URL")]
    controller: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// List models
    List,

    /// Have a model play a game against itself
    Play {
        /// The model ID to use
        #[arg(long, value_name = "ID")]
        model: String,

        /// The number of iters per search. Set to 0 for no search at all
        #[arg(long, value_name = "N", default_value = "1600")]
        iters: usize,

        /// The value decay parameter to use
        #[arg(long, value_name = "F", default_value = "0.0")]
        value_decay: f32,

        /// The sampling temperature to use
        #[arg(long, value_name = "F")]
        temperature: Option<f32>,
    },

    /// Fetch and display recent position buffer items
    Positions {
        /// The model ID to use
        #[arg(long, value_name = "ID")]
        model: String,

        /// The number of recent positions to fetch
        #[arg(long, value_name = "N", default_value = "1")]
        count: usize,
    },

    /// Create a new model
    Create {
        /// The number of conv2d layers to apply after the initial one
        #[arg(long, value_name = "N")]
        conv_layers: usize,

        /// The number of channels in the conv2d layer outputs
        #[arg(long, value_name = "N")]
        conv_channels: usize,

        /// The number of hidden neurons in the value head
        #[arg(long, value_name = "N")]
        value_hidden: usize,
    },
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    let client = {
        let controller_url = cli
            .controller
            .clone()
            .or_else(|| std::env::var("HEX_TRAIN_CONTROLLER_URL").ok())
            .expect("one of --controller or HEX_TRAIN_CONTROLLER_URL must be set");
        ControllerClient::new(controller_url)
    };

    match cli.command {
        Commands::List => {
            let res = client
                .list_models()
                .unwrap_or_else(TrainError::unrecoverable);
            for model in res.models.iter() {
                println!("{}:", model.id);
                for checkpoint in model.checkpoints.iter() {
                    println!("  {}", checkpoint);
                }
            }
        }

        Commands::Create {
            conv_layers,
            conv_channels,
            value_hidden,
        } => {
            let config = ModelConfig {
                conv_layers,
                conv_channels,
                value_hidden,
            };
            client
                .create_model(config)
                .unwrap_or_else(TrainError::unrecoverable);
        }

        Commands::Positions { model, count } => {
            let (_, data) = client
                .fetch_positions(&model, Some(-(count as isize)))
                .unwrap_or_else(TrainError::unrecoverable);
            let n = data.len() / SERIALIZED_LEN;
            for i in 0..n {
                let i0 = i * SERIALIZED_LEN;
                let i1 = (i + 1) * SERIALIZED_LEN;
                let mut pos = Position::deserialize_from(&data[i0..i1]);
                if i % 2 != 0 {
                    pos.apply_transform(&Transpose);
                }
                for r in 0..11 {
                    for _ in 0..r {
                        print!(" ");
                    }
                    print!("\\");
                    for c in 0..11 {
                        let policy = pos.policy[r * 11 + c];
                        let color = (policy.powf(0.5) * (255.0 - 232.0) + 232.0).round() as usize;
                        let num = format!("{:02}", (policy * 99.0).round() as usize);
                        let circle = "○ ";
                        match pos.board.rc(r, c) {
                            Some(true) => print!("\x1b[31m\x1b[48;5;232m{circle}\x1b[0m"),
                            Some(false) => print!("\x1b[36m\x1b[48;5;232m{circle}\x1b[0m"),
                            None => print!("\x1b[48;5;232;38;5;{color}m{num}\x1b[0m"),
                        }
                    }
                    println!("\\");
                }
            }
        }

        Commands::Play {
            model,
            iters,
            value_decay,
            temperature,
        } => {
            let config = client
                .fetch_config(&model)
                .unwrap_or_else(TrainError::unrecoverable);
            let (_, data) = client
                .fetch_model_data(&model, None)
                .unwrap_or_else(TrainError::unrecoverable)
                .into_data()
                .expect("fetch without etag should return data");
            play(config, data, iters, value_decay, temperature);
        }
    }
}

fn play(
    config: ModelConfig,
    data: Vec<u8>,
    iters: usize,
    value_decay: f32,
    temperature: Option<f32>,
) {
    type B = burn::backend::Wgpu<f32, i32>;
    let device = Default::default();
    let model = config.init::<B>(&device);
    let model = model.load_bytes(data, &device);

    let start = Instant::now();
    let mut board = Bitboard::new();
    let mut rng = rand::rngs::SmallRng::seed_from_u64(12345);
    loop {
        if let Some(win) = board.win() {
            println!("({:?})\n{}", start.elapsed(), BitboardPretty(&board));
            println!("({:?}) {} wins", start.elapsed(), if win { "sente" } else { "gote" });
            return;
        }
        if iters > 0 {
            println!("({:?})\n{}", start.elapsed(), BitboardPretty(&board));
            let out = search(&model, &device, board, 0.0, value_decay, |n: usize| {
                print!("\x1b[G\x1b[K{n}/{iters} {:.1}%", n as f64 * 100.0 / iters as f64);
                std::io::stdout().flush().ok();
                n < iters
            });
            println!();
            if let Some(temp) = temperature {
                let m = out
                    .policy
                    .iter()
                    .map(|x| x.powf(1.0 / temp))
                    .sample_weighted(&mut rand::rng())
                    .expect("policy should not be empty");
                board = board.nth_child(m);
                println!("({:?}) value={}", start.elapsed(), out.values[m]);
            } else {
                board = out.board_best;
                println!("({:?}) value={}", start.elapsed(), out.value_best);
            }
        } else {
            let out = model.eval_one(EvalRequest::new(board), &device);
            let policy = out
                .policy
                .iter()
                .enumerate()
                .map(|(i, x)| if board.nth_child_valid(i) { *x } else { 0.0 });
            let m = if let Some(temp) = temperature {
                policy.map(|x| x.powf(1.0 / temp)).sample_weighted(&mut rng)
            } else {
                policy.map(Finite::from).argmax()
            };
            board = board.nth_child(m.expect("no children"));
        }
    }
}
