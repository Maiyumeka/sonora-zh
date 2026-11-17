//! Bounded-memory stem playback. Selected stems share one playback clock and are summed before analysis.
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};
use crate::audio::{self, AudioCapture, SharedAudio};

fn track_entries(manifest: &serde_json::Value) -> Result<Vec<(String, String)>, String> {
    let tracks = manifest["tracks"].as_array().ok_or("分轨清单无效，请重新导入歌曲")?;
    let mut entries = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for track in tracks {
        let file = track["file"].as_str().ok_or("分轨清单缺少文件名")?;
        let path = Path::new(file);
        if path.components().count() != 1 || !matches!(path.components().next(), Some(std::path::Component::Normal(_)))
            || !file.to_lowercase().ends_with(".wav") || !seen.insert(file.to_lowercase()) {
            return Err("分轨清单包含无效或重复文件".into());
        }
        let label = track["label"].as_str().filter(|s| !s.is_empty()).unwrap_or(file);
        entries.push((file.into(), label.into()));
    }
    if entries.len() > 64 { return Err("分轨数量超过播放器支持的 64 条上限".into()); }
    if entries.len() < 2 || entries[0].0 != "mix.wav" { return Err("分轨清单缺少原曲或分轨".into()); }
    Ok(entries)
}

pub struct Song {
    pub tracks: Vec<Vec<[i16; 2]>>,
    pub names: Vec<String>,
    pub rate: u32,
    pub name: String,
}

impl Song {
    pub fn load(directory: &Path, name: String) -> Result<Self, String> {
        let text = std::fs::read_to_string(directory.join("ready.json")).map_err(|e| format!("无法读取分轨清单：{e}"))?;
        let manifest: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("分轨清单格式错误：{e}"))?;
        let entries = track_entries(&manifest)?;
        let mut tracks = Vec::new();
        let mut names = Vec::new();
        let mut rate = 0;
        let mut length = 0;
        for (file, label) in entries {
            let reader = hound::WavReader::open(directory.join(file)).map_err(|e| format!("读取分轨失败：{e}"))?;
            let spec = reader.spec();
            if spec.channels != 2 || spec.bits_per_sample != 16 || spec.sample_format != hound::SampleFormat::Int {
                return Err("分轨必须为双声道 16 位 WAV".into());
            }
            if rate != 0 && rate != spec.sample_rate { return Err("分轨采样率不一致".into()); }
            rate = spec.sample_rate;
            if reader.duration() > rate * 601 { return Err("歌曲超过 10 分钟限制".into()); }
            let samples: Vec<i16> = reader.into_samples().collect::<Result<_, _>>().map_err(|e| e.to_string())?;
            if samples.len() % 2 != 0 { return Err("分轨声道数据不完整".into()); }
            let frames: Vec<_> = samples.chunks_exact(2).map(|f| [f[0], f[1]]).collect();
            if frames.is_empty() || (length != 0 && frames.len() != length) { return Err("分轨长度不一致".into()); }
            length = frames.len();
            tracks.push(frames);
            names.push(label);
        }
        Ok(Self { tracks, names, rate, name })
    }
    pub fn seconds(&self) -> f64 { self.tracks[0].len() as f64 / self.rate as f64 }
    fn sample(&self, track: usize, position: f64) -> [f32; 2] {
        let frames = &self.tracks[track.min(self.tracks.len() - 1)];
        let i = (position as usize).min(frames.len() - 1);
        let j = (i + 1).min(frames.len() - 1);
        let fraction = (position - position.floor()) as f32;
        std::array::from_fn(|ch| (frames[i][ch] as f32 * (1.0 - fraction) + frames[j][ch] as f32 * fraction) / 32768.0)
    }
}

#[derive(Clone, Copy)]
pub struct Controls {
    pub position: f64,
    pub playing: bool,
    pub tracks: u64,
    pub volume: f32,
}
pub struct PlaybackState {
    position: AtomicU64,
    seek_position: AtomicU64,
    seek_version: AtomicU64,
    playing: AtomicBool,
    tracks: AtomicU64,
    volume: AtomicU32,
}
impl PlaybackState {
    fn new() -> Self {
        Self { position: AtomicU64::new(0), seek_position: AtomicU64::new(0),
            seek_version: AtomicU64::new(0), playing: AtomicBool::new(false),
            tracks: AtomicU64::new(1), volume: AtomicU32::new(0.7f32.to_bits()) }
    }
    pub fn snapshot(&self) -> Controls {
        Controls { position: f64::from_bits(self.position.load(Ordering::Relaxed)),
            playing: self.playing.load(Ordering::Relaxed), tracks: self.tracks.load(Ordering::Relaxed),
            volume: f32::from_bits(self.volume.load(Ordering::Relaxed)) }
    }
    pub fn update(&self, before: Controls, after: Controls) {
        if before.position != after.position {
            self.seek_position.store(after.position.to_bits(), Ordering::Relaxed);
            self.seek_version.fetch_add(1, Ordering::Release);
        }
        if before.playing != after.playing { self.playing.store(after.playing, Ordering::Relaxed); }
        if before.tracks != after.tracks { self.tracks.store(after.tracks, Ordering::Relaxed); }
        if before.volume != after.volume { self.volume.store(after.volume.to_bits(), Ordering::Relaxed); }
    }
}
pub struct Playback {
    pub controls: Arc<PlaybackState>,
    pub song: Arc<Song>,
}

impl Playback {
    pub fn start(song: Arc<Song>) -> Result<(Self, AudioCapture), String> {
        let device = audio::default_output_device().ok_or("未找到播放设备")?;
        let supported = device.default_output_config().map_err(|e| e.to_string())?;
        let format = supported.sample_format();
        let sample_rate = supported.sample_rate();
        let channels = supported.channels();
        let config: cpal::StreamConfig = supported.into();
        let controls = Arc::new(PlaybackState::new());
        let shared = Arc::new(Mutex::new(SharedAudio::default()));
        let errors = Arc::new(Mutex::new(None));
        let stream = match format {
            SampleFormat::F32 => output::<f32>(&device, &config, song.clone(), controls.clone(), shared.clone(), errors.clone()),
            SampleFormat::I16 => output::<i16>(&device, &config, song.clone(), controls.clone(), shared.clone(), errors.clone()),
            SampleFormat::U16 => output::<u16>(&device, &config, song.clone(), controls.clone(), shared.clone(), errors.clone()),
            SampleFormat::I32 => output::<i32>(&device, &config, song.clone(), controls.clone(), shared.clone(), errors.clone()),
            _ => return Err("暂不支持此播放设备的采样格式".into()),
        }?;
        stream.play().map_err(|e| format!("播放设备启动失败：{e}"))?;
        let capture = AudioCapture::from_playback(stream, shared, errors, sample_rate, channels);
        Ok((Self { controls, song }, capture))
    }
}

fn output<T: SizedSample + FromSample<f32>>(
    device: &cpal::Device, config: &cpal::StreamConfig, song: Arc<Song>,
    controls: Arc<PlaybackState>, shared: Arc<Mutex<SharedAudio>>, errors: Arc<Mutex<Option<String>>>,
) -> Result<cpal::Stream, String> {
    let channels = config.channels as usize;
    let step = song.rate as f64 / config.sample_rate as f64;
    // Reuse scratch space. Never wait for the UI's mutex on the audio callback.
    let mut analysis = Vec::with_capacity(8192);
    let mut position = 0.0;
    let mut seek_version = 0;
    let mut mixer = TrackMixer::new(song.tracks.len(), config.sample_rate);
    let mut limiter = PreviewLimiter::new(config.sample_rate);
    device.build_output_stream(config.clone(), move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
        let version = controls.seek_version.load(Ordering::Acquire);
        if version != seek_version {
            position = f64::from_bits(controls.seek_position.load(Ordering::Relaxed));
            seek_version = version;
        }
        let mut playing = controls.playing.load(Ordering::Relaxed);
        let tracks = controls.tracks.load(Ordering::Relaxed);
        let volume = f32::from_bits(controls.volume.load(Ordering::Relaxed));
        analysis.clear();
        for frame in data.chunks_exact_mut(channels) {
            let sample = if playing && position < song.tracks[0].len() as f64 {
                let s = mixer.sample(&song, tracks, position);
                position += step;
                s
            } else {
                if playing { controls.playing.store(false, Ordering::Relaxed); playing = false; }
                [0.0; 2]
            };
            let preview = limiter.apply(sample.map(|s| s * volume));
            for (ch, value) in frame.iter_mut().enumerate() {
                let output = if channels == 1 { (preview[0] + preview[1]) * 0.5 } else if ch < 2 { preview[ch] } else { 0.0 };
                *value = T::from_sample(output.clamp(-1.0, 1.0));
            }
            analysis.push(sample);
        }
        controls.position.store(position.to_bits(), Ordering::Relaxed);
        if let Ok(mut ring) = shared.try_lock() { ring.push(&analysis); }
    }, move |error| {
        if let Ok(mut slot) = errors.lock() { *slot = Some(format!("播放中断：{error}")); }
    }, None).map_err(|e| format!("无法打开播放设备：{e}"))
}


/// The original mix is exclusive, so it cannot double the selected stems.
pub fn select_track(mask: u64, index: usize, enabled: bool) -> u64 {
    if index >= 64 { return mask; }
    let bit = 1u64 << index;
    if !enabled { mask & !bit } else if index == 0 { 1 } else { (mask & !1) | bit }
}

struct TrackMixer { gains: Vec<f32>, fade_step: f32 }
impl TrackMixer {
    fn new(tracks: usize, rate: u32) -> Self {
        Self { gains: (0..tracks).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect(),
            fade_step: 1.0 / (rate as f32 * 0.01).max(1.0) }
    }
    fn sample(&mut self, song: &Song, mask: u64, position: f64) -> [f32; 2] {
        let mut mix = [0.0; 2];
        for (i, gain) in self.gains.iter_mut().enumerate() {
            let target = if mask & (1u64 << i) != 0 { 1.0 } else { 0.0 };
            *gain += (target - *gain).clamp(-self.fade_step, self.fade_step);
            if *gain > 0.0 {
                let sample = song.sample(i, position);
                for ch in 0..2 { mix[ch] += sample[ch] * *gain; }
            }
        }
        mix
    }
}

// One stereo-linked gain limits only the preview. Analysis keeps the raw mix.
struct PreviewLimiter { gain: f32, release: f32 }
impl PreviewLimiter {
    fn new(rate: u32) -> Self { Self { gain: 1.0, release: 1.0 - (-1.0 / (rate as f32 * 0.08)).exp() } }
    fn apply(&mut self, sample: [f32; 2]) -> [f32; 2] {
        let peak = sample[0].abs().max(sample[1].abs());
        let target = if peak > 0.98 { 0.98 / peak } else { 1.0 };
        if target < self.gain { self.gain = target; }
        else { self.gain += (target - self.gain) * self.release; }
        sample.map(|s| s * self.gain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn track_list_follows_actual_model_output_including_unknown_labels() {
        let manifest = serde_json::json!({"tracks": [
            {"file": "mix.wav", "label": "原曲"},
            {"file": "voice.wav", "label": "人声"},
            {"file": "custom.wav", "label": "自定义乐器"}
        ]});
        let entries = track_entries(&manifest).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[2], ("custom.wav".into(), "自定义乐器".into()));
        assert!(track_entries(&serde_json::json!({"tracks": [{"file": "../mix.wav"}]})).is_err());
    }

    #[test]
    fn switching_tracks_preserves_source_position_and_resampling() {
        let song = Song { tracks: vec![vec![[0, 0], [16384, -16384]], vec![[8192, 8192], [8192, 8192]]], names: vec!["原曲".into(), "分轨".into()], rate: 44100, name: "test".into() };
        assert_eq!(song.sample(0, 0.5), [0.25, -0.25]);
        assert_eq!(song.sample(1, 0.5), [0.25, 0.25]);
        assert_eq!(song.sample(0, 1.0), [0.5, -0.5]);
    }
    #[test]
    fn controls_only_seek_when_user_changes_the_position() {
        let state = PlaybackState::new();
        let before = state.snapshot();
        let mut after = before;
        after.tracks = 1 << 3;
        after.playing = true;
        state.update(before, after);
        assert_eq!(state.seek_version.load(Ordering::Relaxed), 0);
        assert_eq!(state.snapshot().tracks, 1 << 3);
        after.position = 44100.0;
        state.update(state.snapshot(), after);
        assert_eq!(state.seek_version.load(Ordering::Relaxed), 1);
        assert_eq!(f64::from_bits(state.seek_position.load(Ordering::Relaxed)), 44100.0);
    }
    #[test]
    fn original_and_stems_are_exclusive_but_stems_combine_and_can_all_be_muted() {
        let a = select_track(1, 4, true);
        let b = select_track(a, 5, true);
        assert_eq!(b, (1 << 4) | (1 << 5));
        assert_eq!(select_track(b, 0, true), 1);
        assert_eq!(select_track(select_track(b, 4, false), 5, false), 0);
    }
    #[test]
    fn mixed_stems_use_the_same_fractional_position_and_keep_their_levels() {
        let song = Song { tracks: vec![vec![[30000, 30000]; 2], vec![[0, 0], [16384, -16384]], vec![[8192, 4096]; 2]],
            names: vec!["原曲".into(), "人声".into(), "吉他".into()], rate: 44100, name: "test".into() };
        let mut mixer = TrackMixer::new(3, 48000);
        // Complete the ten-millisecond selection fade at a fixed test position.
        for _ in 0..500 { mixer.sample(&song, 6, 0.5); }
        let sample = mixer.sample(&song, 6, 0.5);
        assert!((sample[0] - 0.5).abs() < 1e-5);
        assert!((sample[1] + 0.125).abs() < 1e-5);
        for _ in 0..500 { mixer.sample(&song, 0, 0.5); }
        assert_eq!(mixer.sample(&song, 0, 0.5), [0.0, 0.0]);
    }
    #[test]
    fn preview_limiter_bounds_peaks_and_preserves_stereo_balance() {
        let mut limiter = PreviewLimiter::new(48000);
        assert_eq!(limiter.apply([0.4, -0.2]), [0.4, -0.2]);
        for sample in [[2.0, -1.0], [3.0, -1.5], [0.6, -0.3]] {
            let result = limiter.apply(sample);
            assert!(result[0].abs().max(result[1].abs()) <= 0.98001);
            assert!((result[0] / result[1] + 2.0).abs() < 1e-5);
        }
    }

}
