## Distributed training

The `train-daemon` binary implements different training workloads as components
that communicate over HTTP:

- `controller` stores and serves model parameters and self play data
- `optimizer` uses self play data to do gradient descent
- `selfplay` uses model parameters to generate self play data

Components are configured with the following environment variables

- Common
  - `RUST_LOG` (optional, but `info` is recommended)
  - `HEX_TRAIN_ROLE` (required)
  - `HEX_TRAIN_CONTROLLER_URL` (required except for controller)
  - `HEX_TRAIN_MODEL_ID` (required except for controller)
- For `HEX_TRAIN_ROLE=controller`
  - `PORT` (default 3000)
  - `HEX_TRAIN_ROOT` (default `data`)
- For `HEX_TRAIN_ROLE=optimizer`
  - `HEX_TRAIN_OPTIMIZER_MAX_POSITIONS` (default 500000)
  - `HEX_TRAIN_OPTIMIZER_UPLOAD_INTERVAL` (seconds, default 300)
  - `HEX_TRAIN_OPTIMIZER_BATCH_SIZE` (default 256)
  - `HEX_TRAIN_OPTIMIZER_MOMENTUM` (default 0.7)
  - `HEX_TRAIN_OPTIMIZER_LEARNING_RATE` (default 0.02)
- For `HEX_TRAIN_ROLE=selfplay`
  - `HEX_TRAIN_SELF_PLAY_BATCH_EVALS` (default 32)
  - `HEX_TRAIN_SELF_PLAY_CONCURRENCY` (default 128)
  - `HEX_TRAIN_SELF_PLAY_ITERS` (default 800)