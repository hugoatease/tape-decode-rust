mod cli;
mod pipeline;
mod stream;
mod writer;

fn main() {
    if let Err(error) = cli::run_cli() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}
