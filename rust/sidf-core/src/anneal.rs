//! Minimal deterministic Model C annealing loop.
//!
//! This is a Rust-core spike, not a bit-perfect port of the current Python
//! NumPy loop. It fixes row-major traversal and uses the crate's Philox helper
//! for initialization, proposal deltas, and accept/reject thresholds.

use crate::energy::{model_c_local_energy, GrayImage, ImageError, ModelCParams};
use crate::prng::{key_from_decoder_seed, philox4x32_10, sidf_counter, Purpose};

const STAGE_INITIALIZE: u32 = 0;
const STAGE_ANNEAL: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnnealConfig {
    pub decoder_seed: u64,
    pub sweeps: usize,
    pub temp_start: f64,
    pub temp_end: f64,
    pub proposal_radius: f64,
}

impl Default for AnnealConfig {
    fn default() -> Self {
        Self {
            decoder_seed: 0,
            sweeps: 40,
            temp_start: 0.5,
            temp_end: 0.01,
            proposal_radius: 0.15,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum DecodeError {
    InvalidConfig(&'static str),
    Image(ImageError),
    TooManyPixels,
}

impl From<ImageError> for DecodeError {
    fn from(value: ImageError) -> Self {
        Self::Image(value)
    }
}

pub fn model_c_decode(
    guide: &GrayImage,
    params: ModelCParams,
    config: AnnealConfig,
) -> Result<GrayImage, DecodeError> {
    validate_config(config)?;
    if guide.width() * guide.height() > u32::MAX as usize {
        return Err(DecodeError::TooManyPixels);
    }

    let key = key_from_decoder_seed(config.decoder_seed);
    let mut state = initialize_state(guide.width(), guide.height(), key)?;

    for sweep in 0..config.sweeps {
        let temp = temperature_at_sweep(config, sweep);
        for pixel_index in 0..guide.width() * guide.height() {
            let row = pixel_index / guide.width();
            let col = pixel_index % guide.width();
            let old_value = state.get(row, col);
            let proposed = clamp01(old_value + proposal_delta(key, sweep, pixel_index, config)?);

            let old_energy = model_c_local_energy(old_value, &state, guide, row, col, params);
            let new_energy = model_c_local_energy(proposed, &state, guide, row, col, params);
            let delta = new_energy - old_energy;

            if should_accept(delta, acceptance_threshold(key, sweep, pixel_index)?, temp) {
                state.set(row, col, proposed);
            }
        }
    }

    Ok(state)
}

fn validate_config(config: AnnealConfig) -> Result<(), DecodeError> {
    if config.sweeps == 0 {
        return Err(DecodeError::InvalidConfig("sweeps must be positive"));
    }
    if !config.temp_start.is_finite() || !config.temp_end.is_finite() {
        return Err(DecodeError::InvalidConfig("temperatures must be finite"));
    }
    if config.temp_start <= 0.0 || config.temp_end <= 0.0 {
        return Err(DecodeError::InvalidConfig("temperatures must be positive"));
    }
    if !config.proposal_radius.is_finite() || config.proposal_radius < 0.0 {
        return Err(DecodeError::InvalidConfig(
            "proposal_radius must be finite and non-negative",
        ));
    }
    Ok(())
}

fn initialize_state(width: usize, height: usize, key: [u32; 2]) -> Result<GrayImage, DecodeError> {
    let mut data = Vec::with_capacity(width * height);
    for pixel_index in 0..width * height {
        let pixel_index = u32::try_from(pixel_index).map_err(|_| DecodeError::TooManyPixels)?;
        let counter = sidf_counter(STAGE_INITIALIZE, 0, pixel_index, Purpose::Proposal);
        data.push(unit_from_word(philox4x32_10(counter, key)[0]));
    }
    Ok(GrayImage::new(width, height, data)?)
}

fn proposal_delta(
    key: [u32; 2],
    sweep: usize,
    pixel_index: usize,
    config: AnnealConfig,
) -> Result<f64, DecodeError> {
    let sweep = u32::try_from(sweep).map_err(|_| DecodeError::TooManyPixels)?;
    let pixel_index = u32::try_from(pixel_index).map_err(|_| DecodeError::TooManyPixels)?;
    let counter = sidf_counter(STAGE_ANNEAL, sweep, pixel_index, Purpose::Proposal);
    let unit = unit_from_word(philox4x32_10(counter, key)[0]);
    Ok(((unit * 2.0) - 1.0) * config.proposal_radius)
}

fn acceptance_threshold(
    key: [u32; 2],
    sweep: usize,
    pixel_index: usize,
) -> Result<f64, DecodeError> {
    let sweep = u32::try_from(sweep).map_err(|_| DecodeError::TooManyPixels)?;
    let pixel_index = u32::try_from(pixel_index).map_err(|_| DecodeError::TooManyPixels)?;
    let counter = sidf_counter(STAGE_ANNEAL, sweep, pixel_index, Purpose::AcceptThreshold);
    Ok(unit_from_word(philox4x32_10(counter, key)[0]))
}

fn should_accept(delta: f64, threshold: f64, temperature: f64) -> bool {
    delta < 0.0 || threshold < (-delta / temperature).exp()
}

fn temperature_at_sweep(config: AnnealConfig, sweep: usize) -> f64 {
    if config.sweeps == 1 {
        return config.temp_start;
    }
    let fraction = sweep as f64 / (config.sweeps - 1) as f64;
    let log_temp =
        config.temp_start.ln() + fraction * (config.temp_end.ln() - config.temp_start.ln());
    log_temp.exp()
}

fn unit_from_word(word: u32) -> f64 {
    f64::from(word) / 4_294_967_296.0
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::{model_c_decode, AnnealConfig};
    use crate::energy::{GrayImage, ModelCParams};

    #[test]
    fn model_c_decode_is_deterministic_for_same_seed() {
        let guide = tiny_cross_guide();
        let params = ModelCParams::default();
        let config = AnnealConfig {
            decoder_seed: 42,
            sweeps: 8,
            temp_start: 0.5,
            temp_end: 0.05,
            proposal_radius: 0.20,
        };

        let first = model_c_decode(&guide, params, config).unwrap();
        let second = model_c_decode(&guide, params, config).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn model_c_decode_keeps_values_in_range() {
        let guide = tiny_cross_guide();
        let decoded = model_c_decode(
            &guide,
            ModelCParams::default(),
            AnnealConfig {
                decoder_seed: 7,
                sweeps: 12,
                temp_start: 0.5,
                temp_end: 0.02,
                proposal_radius: 0.25,
            },
        )
        .unwrap();

        assert!(decoded
            .as_slice()
            .iter()
            .all(|value| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn model_c_decode_moves_tiny_cross_toward_guide() {
        let guide = tiny_cross_guide();
        let decoded = model_c_decode(
            &guide,
            ModelCParams::default(),
            AnnealConfig {
                decoder_seed: 20260530,
                sweeps: 24,
                temp_start: 0.5,
                temp_end: 0.01,
                proposal_radius: 0.20,
            },
        )
        .unwrap();

        let initial_mad = mean_absolute_error(&GrayImage::filled(16, 16, 0.5).unwrap(), &guide);
        let decoded_mad = mean_absolute_error(&decoded, &guide);
        assert!(decoded_mad < initial_mad);
    }

    fn tiny_cross_guide() -> GrayImage {
        let mut data = vec![0.0; 16 * 16];
        for row in 0..16 {
            for col in 0..16 {
                if (6..10).contains(&row) || (6..10).contains(&col) {
                    data[row * 16 + col] = 0.5;
                }
            }
        }
        GrayImage::new(16, 16, data).unwrap()
    }

    fn mean_absolute_error(left: &GrayImage, right: &GrayImage) -> f64 {
        assert_eq!(left.width(), right.width());
        assert_eq!(left.height(), right.height());
        left.as_slice()
            .iter()
            .zip(right.as_slice())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / left.as_slice().len() as f64
    }
}
