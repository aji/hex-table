use std::{io::Write, time::Instant};

use clap::{Parser, Subcommand};
use hex_table::{
    bb::{Bitboard, BitboardPretty},
    nn::{
        model::ModelConfig,
        search::search,
        train::{controller::ControllerClient, error::TrainError},
    },
};

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

        /// The number of iters per search
        #[arg(long, value_name = "N", default_value = "1600")]
        iters: usize,
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

        Commands::Play { model, iters } => {
            let config = client
                .fetch_config(&model)
                .unwrap_or_else(TrainError::unrecoverable);
            let (_, data) = client
                .fetch_model_data(&model, None)
                .unwrap_or_else(TrainError::unrecoverable)
                .into_data()
                .expect("fetch without etag should return data");
            play(config, data, iters);
        }
    }
}

fn play(config: ModelConfig, data: Vec<u8>, iters: usize) {
    type B = burn::backend::Wgpu<f32, i32>;
    let device = Default::default();
    let model = config.init::<B>(&device);
    let model = model.load_bytes(data, &device);

    let start = Instant::now();
    let mut board = Bitboard::new();
    loop {
        if let Some(win) = board.win() {
            println!("({:?})\n{}", start.elapsed(), BitboardPretty(&board));
            println!("({:?}) {} wins", start.elapsed(), if win { "sente" } else { "gote" });
            return;
        }
        println!("({:?})\n{}", start.elapsed(), BitboardPretty(&board));
        let out = search(&model, &device, board, 0.0, |n: usize| {
            print!("\x1b[G\x1b[K{n}/{iters} {:.1}%", n as f64 * 100.0 / iters as f64);
            std::io::stdout().flush().ok();
            n < iters
        });
        println!();
        board = out.board_best;
        println!("({:?}) value={}", start.elapsed(), out.value_best);
    }
}
