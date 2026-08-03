mod cli;
mod decode;
mod fields_match;
mod metadata;
mod profiles;
mod scan;
mod trim;
mod writer;

fn main() {
    if let Err(error) = cli::run_cli() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
