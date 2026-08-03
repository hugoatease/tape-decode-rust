//! Command-line front end: argument parsing and wiring into
//! `crate::pipeline::decode`. Mirrors `vhsdecode/hifi/main.py`'s argument
//! groups and defaults (verified against that source directly), and
//! `tape-decode-cli`'s conventions (kebab-case long flags, `-` for
//! stdin/stdout, `--overwrite` gating output clobbering).
//!
//! Deliberately not implemented, and rejected at runtime with a clear
//! error rather than silently ignored: `--demod hilbert` (only quadrature
//! demod is ported), `--bias_guess`/`--auto_fine_tune` (carrier
//! auto-tracking isn't ported), `--NR_spectral_amount` other than 0 (out
//! of scope for this port, see the plan). Not exposed at all: `--preview`,
//! `--gui`, `--gnuradio`, `--normalize`, `--threads` (this CLI has no
//! concurrency yet — see `pipeline`'s doc comment), and `.ldf`/`.lds`
//! input (those delegate to external tools in Python).

use std::fs::OpenOptions;
use std::path::PathBuf;

use anyhow::{bail, Context as _, Result};
use clap::{Parser, ValueEnum};
use hifi_decode::{AfeOverrides, DecodeMode, EnvDetection, PostProcessParams, ResamplerQuality, System, TapeFormat};
use tape_rf_io::SampleFormat;

use crate::pipeline::{DemodType, DocMode, PipelineParams};
use crate::writer::{write_flac, write_wav};

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum OnOff {
    On,
    Off,
}
impl OnOff {
    fn is_on(self) -> bool {
        self == OnOff::On
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliInputFormat {
    U8,
    S8,
    U16le,
    S16le,
    U10le,
    S10le,
    U12le,
    S12le,
    F32le,
    Flac,
}
impl From<CliInputFormat> for SampleFormat {
    fn from(value: CliInputFormat) -> Self {
        match value {
            CliInputFormat::U8 => SampleFormat::U8,
            CliInputFormat::S8 => SampleFormat::S8,
            CliInputFormat::U16le => SampleFormat::U16LE,
            CliInputFormat::S16le => SampleFormat::S16LE,
            CliInputFormat::U10le => SampleFormat::U10LE,
            CliInputFormat::S10le => SampleFormat::S10LE,
            CliInputFormat::U12le => SampleFormat::U12LE,
            CliInputFormat::S12le => SampleFormat::S12LE,
            CliInputFormat::F32le => SampleFormat::F32LE,
            CliInputFormat::Flac => SampleFormat::Flac,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum CliAudioMode {
    S,
    Ms,
    D,
    Dms,
    L,
    R,
    Sum,
}
impl From<CliAudioMode> for DecodeMode {
    fn from(value: CliAudioMode) -> Self {
        match value {
            CliAudioMode::S => DecodeMode::Stereo,
            CliAudioMode::Ms => DecodeMode::StereoMs,
            CliAudioMode::D => DecodeMode::DualMono,
            CliAudioMode::Dms => DecodeMode::DualMonoMs,
            CliAudioMode::L => DecodeMode::MonoL,
            CliAudioMode::R => DecodeMode::MonoR,
            CliAudioMode::Sum => DecodeMode::MonoSum,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum CliResamplerQuality {
    High,
    Medium,
    Low,
}
impl From<CliResamplerQuality> for ResamplerQuality {
    fn from(value: CliResamplerQuality) -> Self {
        match value {
            CliResamplerQuality::High => ResamplerQuality::High,
            CliResamplerQuality::Medium => ResamplerQuality::Medium,
            CliResamplerQuality::Low => ResamplerQuality::Low,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum CliDocMode {
    Full,
    Mute,
    Off,
}
impl From<CliDocMode> for DocMode {
    fn from(value: CliDocMode) -> Self {
        match value {
            CliDocMode::Full => DocMode::Full,
            CliDocMode::Mute => DocMode::Mute,
            CliDocMode::Off => DocMode::Disabled,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum CliEnvDetection {
    Peak,
    Rms,
}
impl From<CliEnvDetection> for EnvDetection {
    fn from(value: CliEnvDetection) -> Self {
        match value {
            CliEnvDetection::Peak => EnvDetection::Peak,
            CliEnvDetection::Rms => EnvDetection::Rms,
        }
    }
}

/// VHS taus (IEC 60774-2); `HiFiDecode/constants.py`'s
/// `DEFAULT_VHS_*`/`DEFAULT_8MM_*` groups.
struct FormatDefaults {
    audio_mode: CliAudioMode,
    expander_gain: f64,
    expander_ratio: f64,
    expander_attack_tau: f64,
    expander_hold_tau: f64,
    expander_release_tau: f64,
    expander_weighting_low_tau: f64,
    expander_weighting_high_tau: f64,
    deemphasis_low_tau: f64,
    deemphasis_high_tau: f64,
    nr_deemphasis_low_tau: f64,
    nr_deemphasis_high_tau: f64,
}
const VHS_DEFAULTS: FormatDefaults = FormatDefaults {
    audio_mode: CliAudioMode::S,
    expander_gain: 30.0,
    expander_ratio: 2.0,
    expander_attack_tau: 6.5e-3,
    expander_hold_tau: 0.0,
    expander_release_tau: 70e-3,
    expander_weighting_low_tau: 240e-6,
    expander_weighting_high_tau: 24e-6,
    deemphasis_low_tau: 56e-6,
    deemphasis_high_tau: 20e-6,
    nr_deemphasis_low_tau: 240e-6,
    nr_deemphasis_high_tau: 56e-6,
};
const EIGHT_MM_DEFAULTS: FormatDefaults = FormatDefaults {
    audio_mode: CliAudioMode::Ms,
    expander_gain: 6.0,
    expander_ratio: 2.0,
    expander_attack_tau: 3e-3,
    expander_hold_tau: 15e-3,
    expander_release_tau: 40e-3,
    expander_weighting_low_tau: 75e-6,
    expander_weighting_high_tau: 27e-6,
    deemphasis_low_tau: 75e-6,
    deemphasis_high_tau: 27e-6,
    nr_deemphasis_low_tau: 75e-6,
    nr_deemphasis_high_tau: 19e-6,
};

#[derive(Parser, Debug)]
#[command(name = "hifi-decode")]
#[command(about = "Extracts audio from RAW HiFi FM RF captures")]
pub struct Cli {
    /// Source file path, or `-` for stdin (requires --input-format).
    infile: PathBuf,
    /// Output file. `.wav` writes 16-bit PCM; anything else writes 24-bit
    /// FLAC. In `d`/`dms` audio mode, two mono files are written instead:
    /// `<stem>_channel_1<ext>` and `<stem>_channel_2<ext>`.
    outfile: PathBuf,

    /// RF sampling frequency of `infile`. Accepts a bare number (MHz),
    /// or a value with a hz/khz/mhz suffix.
    #[arg(long, short = 'f', default_value = "40")]
    frequency: String,
    /// Allow overwriting an existing output file.
    #[arg(long)]
    overwrite: bool,
    /// Input sample encoding. Required for stdin.
    #[arg(long = "input-format", value_enum)]
    input_format: Option<CliInputFormat>,
    /// Enable debug-level logging unless RUST_LOG supplies an explicit filter.
    #[arg(long)]
    debug: bool,

    // --- System options ---
    /// Source is PAL.
    #[arg(long, short = 'p', group = "system", conflicts_with = "ntsc")]
    pal: bool,
    /// Source is NTSC.
    #[arg(long, short = 'n', group = "system")]
    ntsc: bool,
    /// Use Video8/Hi8 AFM settings instead of VHS HiFi.
    #[arg(long = "8mm")]
    format_8mm: bool,

    // --- Demodulation options ---
    /// FM demodulation type. Only `quadrature` (the default) is
    /// implemented in this port.
    #[arg(long = "demod", default_value = "quadrature")]
    demod_type: String,
    /// Not ported; passing `on` is an error.
    #[arg(long = "bias_guess", alias = "bg")]
    bias_guess: bool,
    /// Not ported; passing `on` is an error.
    #[arg(long = "auto_fine_tune", default_value = "off")]
    auto_fine_tune: String,
    #[arg(long = "AFE_left_carrier", default_value = "0")]
    afe_left_carrier: String,
    #[arg(long = "AFE_left_carrier_deviation", default_value = "0")]
    afe_left_carrier_deviation: String,
    #[arg(long = "AFE_right_carrier", default_value = "0")]
    afe_right_carrier: String,
    #[arg(long = "AFE_right_carrier_deviation", default_value = "0")]
    afe_right_carrier_deviation: String,

    // --- Audio processing options ---
    /// Manual output gain multiplier.
    #[arg(long, default_value_t = 1.0)]
    gain: f64,
    /// Audio channel mode. Defaults to `s` for VHS, `ms` for `--8mm`.
    #[arg(long = "audio_mode", value_enum)]
    audio_mode: Option<CliAudioMode>,
    /// Output sample rate in Hz.
    #[arg(long = "audio_rate", alias = "ar", default_value_t = 48_000)]
    audio_rate: u32,
    /// Resampling quality/speed trade-off.
    #[arg(long, value_enum, default_value = "high")]
    resampler_quality: CliResamplerQuality,

    // --- Noise reduction options ---
    #[arg(long = "head_switching_interpolation", value_enum, default_value = "on")]
    head_switching_interpolation: OnOff,
    #[arg(long = "doc", value_enum, default_value = "full")]
    doc: CliDocMode,
    /// Not ported; only 0 (off) is accepted.
    #[arg(long = "NR_spectral_amount", default_value_t = 0.0)]
    nr_spectral_amount: f64,

    // --- Expander tuning ---
    #[arg(long = "expander", value_enum, default_value = "on")]
    enable_expander: OnOff,
    #[arg(long)]
    expander_gain: Option<f64>,
    #[arg(long)]
    expander_ratio: Option<f64>,
    #[arg(long, value_enum)]
    expander_env_detection: Option<CliEnvDetection>,
    #[arg(long)]
    expander_attack_tau: Option<f64>,
    #[arg(long)]
    expander_hold_tau: Option<f64>,
    #[arg(long)]
    expander_release_tau: Option<f64>,
    #[arg(long)]
    expander_weighting_low_tau: Option<f64>,
    #[arg(long)]
    expander_weighting_high_tau: Option<f64>,

    // --- Deemphasis tuning ---
    #[arg(long = "deemphasis", value_enum, default_value = "on")]
    enable_deemphasis: OnOff,
    #[arg(long)]
    deemphasis_low_tau: Option<f64>,
    #[arg(long)]
    deemphasis_high_tau: Option<f64>,
    #[arg(long)]
    nr_deemphasis_low_tau: Option<f64>,
    #[arg(long)]
    nr_deemphasis_high_tau: Option<f64>,
}

/// `parse_frequency` (`main.py:142-149`, and `tape-decode-cli`'s own
/// equivalent): bare number or hz/khz/mhz-suffixed, returned in MHz.
fn parse_frequency_mhz(value: &str) -> Result<f64> {
    let suffix_start = value
        .find(|ch: char| !matches!(ch, '0'..='9' | '.' | '-' | '+' | 'e' | 'E'))
        .unwrap_or(value.len());
    let (number, suffix) = value.split_at(suffix_start);
    let base: f64 = number.parse().context("invalid frequency value")?;
    let multiplier = match suffix.to_ascii_lowercase().as_str() {
        "" | "m" | "mhz" => 1.0,
        "k" | "khz" => 1.0e-3,
        "hz" => 1.0e-6,
        other => bail!("unknown frequency suffix: {other}"),
    };
    Ok(base * multiplier)
}

pub fn run_cli() -> Result<()> {
    let cli = Cli::parse();
    tape_rf_io::init_tracing(cli.debug);

    if cli.bias_guess {
        bail!("--bias_guess is not implemented in this port");
    }
    if cli.auto_fine_tune.eq_ignore_ascii_case("on") {
        bail!("--auto_fine_tune is not implemented in this port");
    }
    if cli.demod_type.eq_ignore_ascii_case("hilbert") {
        bail!("--demod hilbert is not implemented in this port; only quadrature is available");
    } else if !cli.demod_type.eq_ignore_ascii_case("quadrature") {
        bail!("unknown --demod value: {} (expected quadrature)", cli.demod_type);
    }
    if cli.nr_spectral_amount != 0.0 {
        bail!("--NR_spectral_amount is not implemented in this port; only 0 is accepted");
    }
    if cli.pal && cli.ntsc {
        bail!("--pal and --ntsc are mutually exclusive");
    }

    let system = if cli.ntsc { System::Ntsc } else { System::Pal };
    let format = if cli.format_8mm { TapeFormat::Video8 } else { TapeFormat::Vhs };
    let defaults = if cli.format_8mm { &EIGHT_MM_DEFAULTS } else { &VHS_DEFAULTS };

    let frequency_mhz = parse_frequency_mhz(&cli.frequency)?;
    let afe_overrides = AfeOverrides {
        l_carrier: parse_frequency_mhz(&cli.afe_left_carrier)? * 1.0e6,
        r_carrier: parse_frequency_mhz(&cli.afe_right_carrier)? * 1.0e6,
        l_carrier_deviation: parse_frequency_mhz(&cli.afe_left_carrier_deviation)? * 1.0e6,
        r_carrier_deviation: parse_frequency_mhz(&cli.afe_right_carrier_deviation)? * 1.0e6,
    };

    let mode: DecodeMode = cli.audio_mode.unwrap_or(defaults.audio_mode).into();

    let post_process = PostProcessParams {
        deemphasis_low_tau: cli.deemphasis_low_tau.unwrap_or(defaults.deemphasis_low_tau),
        deemphasis_high_tau: cli.deemphasis_high_tau.unwrap_or(defaults.deemphasis_high_tau),
        nr_deemphasis_low_tau: cli.nr_deemphasis_low_tau.unwrap_or(defaults.nr_deemphasis_low_tau),
        nr_deemphasis_high_tau: cli.nr_deemphasis_high_tau.unwrap_or(defaults.nr_deemphasis_high_tau),
        expander_gain: cli.expander_gain.unwrap_or(defaults.expander_gain),
        expander_ratio: cli.expander_ratio.unwrap_or(defaults.expander_ratio),
        expander_env_detection: cli.expander_env_detection.map(EnvDetection::from).unwrap_or(EnvDetection::Peak),
        expander_attack_tau: cli.expander_attack_tau.unwrap_or(defaults.expander_attack_tau),
        expander_hold_tau: cli.expander_hold_tau.unwrap_or(defaults.expander_hold_tau),
        expander_release_tau: cli.expander_release_tau.unwrap_or(defaults.expander_release_tau),
        expander_weighting_low_tau: cli.expander_weighting_low_tau.unwrap_or(defaults.expander_weighting_low_tau),
        expander_weighting_high_tau: cli.expander_weighting_high_tau.unwrap_or(defaults.expander_weighting_high_tau),
    };

    let params = PipelineParams {
        input_rate: frequency_mhz * 1.0e6,
        format,
        system,
        afe_overrides,
        demod_type: DemodType::Quadrature,
        resampler_quality: cli.resampler_quality.into(),
        audio_final_rate: cli.audio_rate as f64,
        gain: cli.gain,
        mode,
        head_switching_interpolation: cli.head_switching_interpolation.is_on(),
        doc_mode: cli.doc.into(),
        enable_deemphasis: cli.enable_deemphasis.is_on(),
        enable_expander: cli.enable_expander.is_on(),
        post_process,
    };

    let input_format = match cli.input_format {
        Some(f) => f.into(),
        None => {
            if cli.infile.as_os_str() == "-" {
                bail!("--input-format is required when reading from stdin");
            }
            match cli.infile.extension().and_then(|e| e.to_str()) {
                Some(ext) if ext.eq_ignore_ascii_case("flac") => SampleFormat::Flac,
                _ => bail!(
                    "cannot infer input format from {}; pass --input-format explicitly",
                    cli.infile.display()
                ),
            }
        }
    };

    tracing::info!(
        "decoding {} ({:?}, {} MHz, {:?}) -> {}",
        cli.infile.display(),
        format,
        frequency_mhz,
        system,
        cli.outfile.display()
    );

    let input_file = if cli.infile.as_os_str() == "-" {
        tape_rf_io::stdin_file()?
    } else {
        OpenOptions::new()
            .read(true)
            .open(&cli.infile)
            .with_context(|| format!("failed to open input {}", cli.infile.display()))?
    };
    let mut reader = tape_rf_io::DecodeReader::new(tape_rf_io::open_source(input_file, input_format)?);

    let mut rf = Vec::new();
    let mut chunk = vec![0.0f32; 1 << 20];
    loop {
        let n = reader.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        rf.extend_from_slice(&chunk[..n]);
    }
    if rf.is_empty() {
        bail!("no input samples read from {}", cli.infile.display());
    }
    tracing::info!("read {} RF samples", rf.len());

    let (left, right) = crate::pipeline::decode(&rf, &params)?;

    write_output(&cli.outfile, params.audio_final_rate as u32, mode, &left, &right, cli.overwrite)?;
    tracing::info!("wrote {}", cli.outfile.display());
    Ok(())
}

fn write_output(outfile: &PathBuf, sample_rate: u32, mode: DecodeMode, left: &[f32], right: &[f32], overwrite: bool) -> Result<()> {
    let is_wav = outfile.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("wav"));

    if !overwrite {
        check_does_not_exist(outfile)?;
    }

    if mode == DecodeMode::DualMono || mode == DecodeMode::DualMonoMs {
        let (path1, path2) = dual_mono_paths(outfile);
        if !overwrite {
            check_does_not_exist(&path1)?;
            check_does_not_exist(&path2)?;
        }
        if is_wav {
            write_wav(&path1, sample_rate, left, None)?;
            write_wav(&path2, sample_rate, right, None)?;
        } else {
            write_flac(&path1, sample_rate, left, None)?;
            write_flac(&path2, sample_rate, right, None)?;
        }
    } else if is_wav {
        write_wav(outfile, sample_rate, left, Some(right))?;
    } else {
        write_flac(outfile, sample_rate, left, Some(right))?;
    }
    Ok(())
}

fn check_does_not_exist(path: &std::path::Path) -> Result<()> {
    if path.exists() {
        bail!("output {} already exists; pass --overwrite to replace it", path.display());
    }
    Ok(())
}

/// `<stem>_channel_1<ext>` / `<stem>_channel_2<ext>` (`main.py`'s
/// `get_dual_mono_filename`).
fn dual_mono_paths(outfile: &std::path::Path) -> (PathBuf, PathBuf) {
    let ext = outfile.extension().and_then(|e| e.to_str()).unwrap_or("");
    let stem = outfile.with_extension("");
    let stem = stem.to_string_lossy();
    if ext.is_empty() {
        (PathBuf::from(format!("{stem}_channel_1")), PathBuf::from(format!("{stem}_channel_2")))
    } else {
        (PathBuf::from(format!("{stem}_channel_1.{ext}")), PathBuf::from(format!("{stem}_channel_2.{ext}")))
    }
}
