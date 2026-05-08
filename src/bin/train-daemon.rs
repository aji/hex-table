use hex_table::nn::train;

fn main() {
    env_logger::init();
    let role = std::env::var("HEX_TRAIN_ROLE").expect("HEX_TRAIN_ROLE is a required env var");
    match role.as_str() {
        "controller" => train::controller::main(),
        "optimizer" => train::optimizer::main(),
        "selfplay" => train::selfplay::main(),
        _ => panic!("unknown role: {role}"),
    }
}
