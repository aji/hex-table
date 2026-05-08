FROM rust:1-slim-trixie AS builder
WORKDIR /usr/src/hex-table
COPY . .
RUN cargo install --path . --bin train-daemon

FROM debian:trixie-slim
COPY --from=builder /usr/local/cargo/bin/train-daemon /usr/local/bin/train-daemon
CMD ["train-daemon"]