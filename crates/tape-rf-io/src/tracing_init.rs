use std::io;

/// Initialize the shared `tracing` subscriber for a CLI binary: writes to
/// stderr, defaults to `info` (or `debug` when `debug` is set), and honors
/// `RUST_LOG` when present. Safe to call more than once per process; only the
/// first call takes effect.
pub fn init_tracing(debug: bool) {
    let filter = if debug { "debug" } else { "info" };
    let _ = tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| filter.into()),
        )
        .try_init();
}
