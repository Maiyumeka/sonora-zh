//! Compare cached stems at one source-frame position, without running more models.
use std::sync::Arc;
use std::time::{Duration, Instant};
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use crate::analysis::{FFT_SIZE, SPECTRUM_BINS};
use crate::player::Song;

#[derive(Clone)]
pub struct Layer {
    pub track: usize,
    pub spectrum_db: Vec<f32>,
}

#[derive(Clone)]
pub struct ComparisonView {
    pub sample_rate: u32,
    pub layers: Vec<Layer>,
    pub message: Option<&'static str>,
}

pub struct SpectralComparison {
    pub enabled: Vec<bool>,
    layers: Vec<Layer>,
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    norm: f32,
    scratch: Vec<Complex<f32>>,
    last_update: Option<Instant>,
    last_position: Option<usize>,
    last_enabled: Vec<bool>,
}

impl SpectralComparison {
    pub fn new(tracks: usize) -> Self {
        let window: Vec<f32> = (0..FFT_SIZE).map(|i| {
            0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / FFT_SIZE as f32).cos()
        }).collect();
        Self {
            enabled: (0..tracks).map(|i| i != 0).collect(),
            layers: (0..tracks).map(|track| Layer { track, spectrum_db: vec![-120.0; SPECTRUM_BINS] }).collect(),
            fft: FftPlanner::new().plan_fft_forward(FFT_SIZE),
            norm: 2.0 / window.iter().sum::<f32>(), window,
            scratch: vec![Complex::new(0.0, 0.0); FFT_SIZE],
            last_update: None, last_position: None, last_enabled: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers { layer.spectrum_db.fill(-120.0); }
        self.last_update = None;
        self.last_position = None;
    }

    pub fn update(&mut self, song: &Song, position: f64) -> bool {
        let end = (position.max(0.0) as usize).min(song.tracks[0].len());
        let selection_changed = self.last_enabled != self.enabled;
        if !selection_changed && self.last_position == Some(end) { return false; }
        let now = Instant::now();
        let elapsed = self.last_update.map(|t| now.duration_since(t));
        if !selection_changed && elapsed.is_some_and(|t| t < Duration::from_millis(33)) { return false; }
        if self.last_position.is_some_and(|old| old.abs_diff(end) > song.rate as usize / 4) {
            self.clear();
        }
        let release = elapsed.map(|d| 1.0 - (-d.as_secs_f32() / 0.18).exp()).unwrap_or(1.0);
        self.compute(song, end, release);
        self.last_position = Some(end);
        self.last_update = Some(now);
        self.last_enabled.clone_from(&self.enabled);
        true
    }

    fn compute(&mut self, song: &Song, end: usize, release: f32) {
        for layer in &mut self.layers {
            if !self.enabled[layer.track] { continue; }
            let frames = &song.tracks[layer.track];
            for (i, value) in self.scratch.iter_mut().enumerate() {
                let offset = end as i64 - FFT_SIZE as i64 + i as i64;
                let mono = if offset >= 0 {
                    frames.get(offset as usize).map(|f| (f[0] as f32 + f[1] as f32) / 65536.0).unwrap_or(0.0)
                } else { 0.0 };
                *value = Complex::new(mono * self.window[i], 0.0);
            }
            self.fft.process(&mut self.scratch);
            for (i, previous) in layer.spectrum_db.iter_mut().enumerate() {
                let db = 20.0 * (self.scratch[i].norm() * self.norm).max(1e-9).log10();
                *previous = if db >= *previous { db } else { *previous + (db - *previous) * release };
            }
        }
    }

    pub fn view(&self, sample_rate: u32) -> ComparisonView {
        ComparisonView { sample_rate, message: None, layers: self.layers.iter().filter(|l| self.enabled[l.track]).cloned().collect() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tone(bin: usize, amplitude: f32) -> Vec<[i16; 2]> {
        (0..FFT_SIZE * 2).map(|i| {
            let value = (amplitude * (2.0 * std::f32::consts::PI * bin as f32 * i as f32 / FFT_SIZE as f32).sin() * 32767.0) as i16;
            [value, value]
        }).collect()
    }
    fn song(a: Vec<[i16; 2]>, b: Vec<[i16; 2]>) -> Song {
        Song { tracks: vec![a.clone(), a, b], names: vec!["原曲".into(), "乐器一".into(), "乐器二".into()], rate: 48000, name: "test".into() }
    }
    #[test]
    fn simultaneous_layers_keep_distinct_frequencies_on_one_clock() {
        let song = song(tone(32, 0.4), tone(96, 0.4));
        let mut comparison = SpectralComparison::new(3);
        comparison.compute(&song, FFT_SIZE * 2, 1.0);
        let view = comparison.view(song.rate);
        let peak = |l: &Layer| l.spectrum_db.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1)).unwrap().0;
        assert_eq!(view.layers.len(), 2);
        assert_eq!(peak(&view.layers[0]), 32);
        assert_eq!(peak(&view.layers[1]), 96);
    }
    #[test]
    fn comparison_preserves_relative_levels_instead_of_normalizing_each_track() {
        let song = song(tone(32, 0.4), tone(32, 0.2));
        let mut comparison = SpectralComparison::new(3);
        comparison.compute(&song, FFT_SIZE * 2, 1.0);
        let view = comparison.view(song.rate);
        let delta = view.layers[0].spectrum_db[32] - view.layers[1].spectrum_db[32];
        assert!((delta - 6.02).abs() < 0.05);
    }
    #[test]
    fn paused_comparison_is_reused_and_checkbox_changes_take_effect() {
        let song = song(tone(32, 0.4), tone(96, 0.4));
        let mut comparison = SpectralComparison::new(3);
        assert!(comparison.update(&song, 8192.0));
        assert!(!comparison.update(&song, 8192.0));
        comparison.enabled[1] = false;
        assert!(comparison.update(&song, 8192.0));
        assert_eq!(comparison.view(song.rate).layers.len(), 1);
    }
}
