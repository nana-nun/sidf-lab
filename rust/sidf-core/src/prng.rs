//! Counter-based PRNG helpers for the Rust core spike.
//!
//! The first target is `Philox4x32-10`, matching
//! `references/prng-test-vectors/philox4x32-10.json`.
//! This module fixes integer output words only; it does not define proposal
//! sampling, accept/reject thresholds, fixed-point arithmetic, or decoder state.

const M0: u32 = 0xD251_1F53;
const M1: u32 = 0xCD9E_8D57;
const W0: u32 = 0x9E37_79B9;
const W1: u32 = 0xBB67_AE85;
const ROUNDS: usize = 10;

/// Random-use selector stored in `counter[3]`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum Purpose {
    Proposal = 0,
    AcceptThreshold = 1,
    TexturePrior = 2,
    Reserved = 3,
}

/// Build the minimal SIDF counter layout used by the saved spike vectors.
///
/// Word order:
///
/// - `counter[0]`: stage
/// - `counter[1]`: sweep
/// - `counter[2]`: row-major pixel index
/// - `counter[3]`: purpose selector
pub const fn sidf_counter(stage: u32, sweep: u32, pixel_index: u32, purpose: Purpose) -> [u32; 4] {
    [stage, sweep, pixel_index, purpose as u32]
}

/// Split the current 64-bit decoder-seed convention into two Philox key words.
///
/// This is a test-vector convention, not a final SIDF seed expansion design.
pub const fn key_from_decoder_seed(decoder_seed: u64) -> [u32; 2] {
    [(decoder_seed >> 32) as u32, decoder_seed as u32]
}

/// Evaluate `Philox4x32-10` with wrapping `u32` arithmetic.
///
/// Output word order is `[word0, word1, word2, word3]` after the final round.
pub fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut counter = counter;
    let mut key = key;

    for _ in 0..ROUNDS {
        counter = philox4x32_round(counter, key);
        key = [key[0].wrapping_add(W0), key[1].wrapping_add(W1)];
    }

    counter
}

fn philox4x32_round(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (hi0, lo0) = mulhilo(M0, counter[0]);
    let (hi1, lo1) = mulhilo(M1, counter[2]);

    [
        hi1 ^ counter[1] ^ key[0],
        lo1,
        hi0 ^ counter[3] ^ key[1],
        lo0,
    ]
}

fn mulhilo(multiplier: u32, value: u32) -> (u32, u32) {
    let product = u64::from(multiplier) * u64::from(value);
    ((product >> 32) as u32, product as u32)
}

#[cfg(test)]
mod tests {
    use super::{key_from_decoder_seed, philox4x32_10, sidf_counter, Purpose};

    #[test]
    fn saved_vector_zero_matches() {
        assert_eq!(
            philox4x32_10([0, 0, 0, 0], [0, 0]),
            [0x6627_E8D5, 0xE169_C58D, 0xBC57_AC4C, 0x9B00_DBD8]
        );
    }

    #[test]
    fn saved_vector_seed_1_proposal_matches() {
        assert_eq!(
            philox4x32_10(
                sidf_counter(0, 0, 0, Purpose::Proposal),
                key_from_decoder_seed(0x0000_0000_0000_0001),
            ),
            [0xFDDE_3E0B, 0xFA7E_58B6, 0x3380_EC46, 0xD8D5_5C4F]
        );
    }

    #[test]
    fn saved_vector_seed_1_accept_matches() {
        assert_eq!(
            philox4x32_10(
                sidf_counter(0, 3, 17, Purpose::AcceptThreshold),
                key_from_decoder_seed(0x0000_0000_0000_0001),
            ),
            [0x8079_032B, 0x2788_7360, 0xA0CC_833F, 0xA668_CBB4]
        );
    }

    #[test]
    fn saved_vector_deadbeef_texture_matches() {
        assert_eq!(
            philox4x32_10(
                sidf_counter(2, 7, 255, Purpose::TexturePrior),
                key_from_decoder_seed(0xDEAD_BEEF_CAFE_BABE),
            ),
            [0x8906_45AD, 0x537D_F807, 0x938C_AD9D, 0x0E66_A283]
        );
    }

    #[test]
    fn saved_vector_12345678_proposal_matches() {
        assert_eq!(
            philox4x32_10(
                sidf_counter(1, 42, 4095, Purpose::Proposal),
                key_from_decoder_seed(0x1234_5678_9ABC_DEF0),
            ),
            [0xB8C5_0FD1, 0x4716_35D6, 0x1A6B_560F, 0x0D65_0605]
        );
    }
}
