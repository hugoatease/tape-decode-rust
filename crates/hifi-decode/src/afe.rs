//! AFE (analog front-end) carrier parameters and bandpass filter. Ports
//! `AFEParams*`/`get_standard`/`AFEFilterable` from `HiFiDecode.py:73-254`.

use sci_rs::signal::filter::design::{FilterBandType, Sos};
use tape_dsp::{cheby2_sos, narrow_sos, sosfiltfilt_f32};

/// Tape format: selects the AFE carrier layout (VHS HiFi vs Video8/Hi8 AFM).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TapeFormat {
    Vhs,
    Video8,
}

/// Color/line system. Only `Vhs` carriers differ between PAL and NTSC (both
/// derived from the same base deviations/notch widths); `Video8` AFE
/// parameters are identical between systems (only the unrelated `Hfreq`
/// differs in Python, which isn't used by the AFE/demod path).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum System {
    Pal,
    Ntsc,
}

/// Per-channel AFE carrier parameters: center frequency, FM deviation, and
/// the bandpass half-width (Carson's-rule-derived for VHS; ad hoc for
/// Video8, matching Python exactly).
#[derive(Clone, Copy, Debug)]
pub struct AfeParams {
    pub l_carrier_ref: f64,
    pub r_carrier_ref: f64,
    pub l_carrier_deviation: f64,
    pub r_carrier_deviation: f64,
    pub l_notch_width: f64,
    pub r_notch_width: f64,
}

/// Optional per-channel carrier/deviation overrides (the CLI's
/// `--AFE_left_carrier` family; `0.0` means "no override", matching
/// Python's `if value != 0` checks in `get_standard`).
#[derive(Clone, Copy, Debug, Default)]
pub struct AfeOverrides {
    pub l_carrier: f64,
    pub r_carrier: f64,
    pub l_carrier_deviation: f64,
    pub r_carrier_deviation: f64,
}

impl AfeParams {
    /// `AFEParamsVHS.__init__` (`HiFiDecode.py:74-82`): Carson's bandwidth
    /// rule with a fixed 35.753125 kHz padding term.
    fn vhs_base(l_carrier_ref: f64, r_carrier_ref: f64) -> Self {
        const DEVIATION: f64 = 150e3;
        const NOTCH_PADDING: f64 = 35.753125e3;
        let notch_width = 2.0 * (DEVIATION + NOTCH_PADDING);
        AfeParams {
            l_carrier_ref,
            r_carrier_ref,
            l_carrier_deviation: DEVIATION,
            r_carrier_deviation: DEVIATION,
            l_notch_width: notch_width,
            r_notch_width: notch_width,
        }
    }

    /// `AFEParams8mm.__init__` (`HiFiDecode.py:112-121`): asymmetric main
    /// (L, ±100kHz) / sub (R, ±50kHz) channel deviations and notch widths.
    /// Identical for PAL and NTSC Video8/Hi8 (only `Hfreq`, unused here,
    /// differs between the two Python subclasses).
    fn video8_base() -> Self {
        const L_DEVIATION: f64 = 100e3;
        const R_DEVIATION: f64 = 50e3;
        AfeParams {
            l_carrier_ref: 1.5e6,
            r_carrier_ref: 1.7e6,
            l_carrier_deviation: L_DEVIATION,
            r_carrier_deviation: R_DEVIATION,
            l_notch_width: 2.0 * (L_DEVIATION + 20e3),
            r_notch_width: 1.5 * R_DEVIATION,
        }
    }

    /// Equivalent of `HiFiDecode.get_standard` (`HiFiDecode.py:155-183`),
    /// minus the field-rate lookup (that lives with block sizing).
    pub fn for_format(format: TapeFormat, system: System, overrides: AfeOverrides) -> Self {
        let mut params = match (format, system) {
            (TapeFormat::Vhs, System::Pal) => AfeParams::vhs_base(1.4e6, 1.8e6),
            (TapeFormat::Vhs, System::Ntsc) => AfeParams::vhs_base(1.3e6, 1.7e6),
            (TapeFormat::Video8, System::Pal | System::Ntsc) => AfeParams::video8_base(),
        };

        if overrides.l_carrier_deviation != 0.0 {
            params.l_carrier_deviation = overrides.l_carrier_deviation;
        }
        if overrides.r_carrier_deviation != 0.0 {
            params.r_carrier_deviation = overrides.r_carrier_deviation;
        }
        if overrides.l_carrier != 0.0 {
            params.l_carrier_ref = overrides.l_carrier;
        }
        if overrides.r_carrier != 0.0 {
            params.r_carrier_ref = overrides.r_carrier;
        }
        params
    }
}

/// Field rate in Hz, matching `get_standard`'s `field_rate` return
/// (`HiFiDecode.py:161,164,168,171`).
pub fn field_rate(system: System) -> f64 {
    match system {
        System::Pal => 50.0,
        System::Ntsc => 59.94,
    }
}

/// Zero-phase Chebyshev II bandpass carrier filter for one audio channel.
/// Ports `AFEFilterable` (`HiFiDecode.py:229-254`): order 22, 220 dB
/// stopband, `[center - width, center + width]`, applied via
/// `sosfiltfilt_rust`/`tape_dsp::sosfiltfilt_f32`.
pub struct AfeFilter {
    sos: Vec<Sos<f32>>,
}

impl AfeFilter {
    pub fn design(carrier_ref: f64, notch_width: f64, sample_rate: f64) -> Self {
        let low = carrier_ref - notch_width;
        let high = carrier_ref + notch_width;
        let sos_f64 = cheby2_sos(22, 220.0, &[low, high], FilterBandType::Bandpass, sample_rate);
        AfeFilter {
            sos: narrow_sos(&sos_f64),
        }
    }

    pub fn work(&self, data: &[f32]) -> Vec<f32> {
        sosfiltfilt_f32(&self.sos, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pal_vhs_matches_python_afe_params() {
        let params = AfeParams::for_format(TapeFormat::Vhs, System::Pal, AfeOverrides::default());
        assert_eq!(params.l_carrier_ref, 1_400_000.0);
        assert_eq!(params.r_carrier_ref, 1_800_000.0);
        assert_eq!(params.l_carrier_deviation, 150_000.0);
        assert_eq!(params.r_carrier_deviation, 150_000.0);
        // 2 * (150e3 + 35.753125e3)
        assert!((params.l_notch_width - 371_506.25).abs() < 1e-6);
        assert!((params.r_notch_width - 371_506.25).abs() < 1e-6);
    }

    #[test]
    fn video8_matches_python_afe_params() {
        let params =
            AfeParams::for_format(TapeFormat::Video8, System::Ntsc, AfeOverrides::default());
        assert_eq!(params.l_carrier_ref, 1_500_000.0);
        assert_eq!(params.r_carrier_ref, 1_700_000.0);
        assert_eq!(params.l_carrier_deviation, 100_000.0);
        assert_eq!(params.r_carrier_deviation, 50_000.0);
        assert_eq!(params.l_notch_width, 240_000.0);
        assert_eq!(params.r_notch_width, 75_000.0);
    }

    #[test]
    fn overrides_replace_only_nonzero_fields() {
        let overrides = AfeOverrides {
            l_carrier: 1_350_000.0,
            ..Default::default()
        };
        let params = AfeParams::for_format(TapeFormat::Vhs, System::Pal, overrides);
        assert_eq!(params.l_carrier_ref, 1_350_000.0);
        assert_eq!(params.r_carrier_ref, 1_800_000.0); // untouched
    }
}
