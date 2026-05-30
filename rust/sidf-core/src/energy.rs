//! Minimal Model C energy helpers.
//!
//! This mirrors the current Python Model C local energy formula while keeping
//! the Rust spike limited to grayscale `f64` state and guide images.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelCParams {
    pub j_base: f64,
    pub lambda_data: f64,
    pub gamma: f64,
}

impl Default for ModelCParams {
    fn default() -> Self {
        Self {
            j_base: 2.0,
            lambda_data: 5.0,
            gamma: 40.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GrayImage {
    width: usize,
    height: usize,
    data: Vec<f64>,
}

impl GrayImage {
    pub fn new(width: usize, height: usize, data: Vec<f64>) -> Result<Self, ImageError> {
        if width == 0 || height == 0 {
            return Err(ImageError::EmptyShape);
        }
        if data.len() != width * height {
            return Err(ImageError::LengthMismatch {
                expected: width * height,
                actual: data.len(),
            });
        }
        if data.iter().any(|value| !value.is_finite()) {
            return Err(ImageError::NonFiniteValue);
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    pub fn filled(width: usize, height: usize, value: f64) -> Result<Self, ImageError> {
        Self::new(width, height, vec![value; width * height])
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[self.index(row, col)]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        let index = self.index(row, col);
        self.data[index] = value;
    }

    pub const fn index(&self, row: usize, col: usize) -> usize {
        row * self.width + col
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ImageError {
    EmptyShape,
    LengthMismatch { expected: usize, actual: usize },
    NonFiniteValue,
}

pub fn valid_neighbors(row: usize, col: usize, height: usize, width: usize) -> Vec<(usize, usize)> {
    let mut neighbors = Vec::with_capacity(4);
    if row > 0 {
        neighbors.push((row - 1, col));
    }
    if row + 1 < height {
        neighbors.push((row + 1, col));
    }
    if col > 0 {
        neighbors.push((row, col - 1));
    }
    if col + 1 < width {
        neighbors.push((row, col + 1));
    }
    neighbors
}

pub fn model_c_local_energy(
    value: f64,
    state: &GrayImage,
    guide: &GrayImage,
    row: usize,
    col: usize,
    params: ModelCParams,
) -> f64 {
    assert_eq!(state.width(), guide.width());
    assert_eq!(state.height(), guide.height());

    let guide_value = guide.get(row, col);
    let fidelity = params.lambda_data * (value - guide_value).powi(2);
    let mut smooth = 0.0;

    for (neighbor_row, neighbor_col) in valid_neighbors(row, col, guide.height(), guide.width()) {
        let neighbor_value = state.get(neighbor_row, neighbor_col);
        let neighbor_guide = guide.get(neighbor_row, neighbor_col);
        let weight = params.j_base * (-params.gamma * (guide_value - neighbor_guide).powi(2)).exp();
        smooth += weight * (value - neighbor_value).powi(2);
    }

    fidelity + smooth
}

#[cfg(test)]
mod tests {
    use super::{model_c_local_energy, valid_neighbors, GrayImage, ModelCParams};

    #[test]
    fn valid_neighbors_corner_matches_python_order() {
        assert_eq!(valid_neighbors(0, 0, 3, 3), vec![(1, 0), (0, 1)]);
    }

    #[test]
    fn model_c_prefers_guide_value_when_neighbors_match() {
        let guide = GrayImage::filled(3, 3, 0.5).unwrap();
        let state = GrayImage::filled(3, 3, 0.5).unwrap();
        let params = ModelCParams {
            j_base: 1.0,
            lambda_data: 5.0,
            gamma: 40.0,
        };
        let at_guide = model_c_local_energy(0.5, &state, &guide, 1, 1, params);
        let away = model_c_local_energy(0.0, &state, &guide, 1, 1, params);
        assert!(at_guide < away);
    }

    #[test]
    fn model_c_energy_matches_tiny_hand_calculation() {
        let guide = GrayImage::new(2, 2, vec![0.25, 0.25, 0.75, 0.75]).unwrap();
        let state = GrayImage::new(2, 2, vec![0.25, 0.50, 0.75, 1.00]).unwrap();
        let params = ModelCParams {
            j_base: 2.0,
            lambda_data: 5.0,
            gamma: 0.0,
        };

        let energy = model_c_local_energy(0.5, &state, &guide, 0, 0, params);

        let fidelity = 5.0_f64 * (0.5_f64 - 0.25_f64).powi(2);
        let smooth_down = 2.0_f64 * (0.5_f64 - 0.75_f64).powi(2);
        let smooth_right = 2.0_f64 * (0.5_f64 - 0.50_f64).powi(2);
        let expected = fidelity + smooth_down + smooth_right;
        assert!((energy - expected).abs() < 1e-12);
    }
}
