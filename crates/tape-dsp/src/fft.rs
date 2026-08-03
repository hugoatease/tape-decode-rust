use realfft::{ComplexToReal, RealToComplex};
use rustfft::num_complex::Complex32;

/// Forward FFT of a real signal to its `n/2 + 1` unique spectrum bins.
/// The r2c transform consumes its input as scratch, so the buffer is taken by
/// value; callers whose signal must survive pass a copy via `rfft_f32`.
pub fn rfft_owned_f32(mut buffer: Vec<f32>, r2c: &dyn RealToComplex<f32>) -> Vec<Complex32> {
    assert_eq!(buffer.len(), r2c.len());
    let mut output = r2c.make_output_vec();
    r2c.process(&mut buffer, &mut output)
        .expect("r2c forward FFT failed");
    output
}

pub fn rfft_f32(input: &[f32], r2c: &dyn RealToComplex<f32>) -> Vec<Complex32> {
    rfft_owned_f32(input.to_vec(), r2c)
}

/// See `rfft_owned_f32` for the ownership contract.
pub fn irfft_owned_f32(
    mut spectrum: Vec<Complex32>,
    n: Option<usize>,
    c2r: &dyn ComplexToReal<f32>,
) -> Vec<f32> {
    if spectrum.is_empty() {
        return Vec::new();
    }

    let n = n.unwrap_or_else(|| 2 * (spectrum.len() - 1));
    assert_eq!(n, c2r.len());
    assert_eq!(spectrum.len(), (n / 2) + 1);

    // The c2r transform rebuilds the signal from the unique bins directly
    // (half-length inner FFT) instead of mirroring out the full Hermitian
    // spectrum and running a full-length complex inverse. It rejects residual
    // imaginary parts on the DC/Nyquist bins, which the filters can leave
    // behind; the full-spectrum path simply dropped them (only the real part
    // was kept), so zero them to match.
    spectrum[0].im = 0.0;
    spectrum[n / 2].im = 0.0;
    let mut output = c2r.make_output_vec();
    c2r.process(&mut spectrum, &mut output)
        .expect("c2r inverse FFT failed");
    let inv_scale = 1.0 / n as f32;
    for sample in &mut output {
        *sample *= inv_scale;
    }
    output
}

pub fn irfft_f32(input: &[Complex32], n: Option<usize>, c2r: &dyn ComplexToReal<f32>) -> Vec<f32> {
    irfft_owned_f32(input.to_vec(), n, c2r)
}
