//! Persistent local model process, bounded latest-window input and latest-result output.
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use std::time::{Duration, Instant};
use crate::comparison::{ComparisonView, SpectralComparison};
use crate::player::Song;
use crate::separation::{self, SeparationModel};

const WINDOW: usize = 6;
const HOP: usize = 2;
const OUTPUT_RATE: usize = 44100;
const PRE_ROLL: usize = crate::analysis::FFT_SIZE;
const RESULT_FRAMES: usize = OUTPUT_RATE * HOP + PRE_ROLL;

pub struct LiveInput {
    frames: VecDeque<[f32; 2]>,
    pub rate: u32,
    total: u64,
    captured: Instant,
}
struct Window { frames: Vec<[f32; 2]>, end: u64, captured: Instant }
impl LiveInput {
    fn new(rate: u32) -> Self { Self { frames: VecDeque::new(), rate, total: 0, captured: Instant::now() } }
    pub fn push(&mut self, frames: &[[f32; 2]]) {
        if frames.is_empty() { return; }
        // A stalled UI or paused loopback can omit time. Never join old and new audio.
        if self.total > 0 && self.captured.elapsed() > Duration::from_millis(750) { self.frames.clear(); }
        self.total += frames.len() as u64;
        self.frames.extend(frames.iter().copied());
        let excess = self.frames.len().saturating_sub(self.rate as usize * WINDOW);
        self.frames.drain(..excess);
        self.captured = Instant::now();
    }
    pub fn seconds(&self) -> f32 { self.frames.len() as f32 / self.rate as f32 }
    fn latest(&self, last: Option<u64>) -> Option<Window> {
        if self.frames.len() != self.rate as usize * WINDOW
            || last.is_some_and(|last| self.total.saturating_sub(last) < self.rate as u64 * HOP as u64) { return None; }
        Some(Window { frames: self.frames.iter().copied().collect(), end: self.total, captured: self.captured })
    }
}

pub struct LiveBlock { pub song: Arc<Song>, pub source_start: Instant }
pub struct LiveState {
    pub message: String,
    pub names: Vec<String>,
    pub latest: Option<LiveBlock>,
    pub error: Option<String>,
    pub inference_secs: f32,
    pub reserved_mb: u64,
    pub skipped: u64,
}
pub struct LiveJob {
    pub input: Arc<Mutex<LiveInput>>,
    pub state: Arc<Mutex<LiveState>>,
    cancelled: Arc<AtomicBool>,
    child: Arc<Mutex<Option<Child>>>,
}
impl LiveJob {
    pub fn start(rate: u32, model: SeparationModel) -> Self {
        let input = Arc::new(Mutex::new(LiveInput::new(rate)));
        let state = Arc::new(Mutex::new(LiveState { message: "加载实时模型并预热显卡…".into(), names: Vec::new(), latest: None,
            error: None, inference_secs: 0.0, reserved_mb: 0, skipped: 0 }));
        let cancelled = Arc::new(AtomicBool::new(false));
        let child = Arc::new(Mutex::new(None));
        let (i, s, c, p) = (input.clone(), state.clone(), cancelled.clone(), child.clone());
        std::thread::spawn(move || {
            if let Err(error) = run(rate, model, &i, &s, &c, &p) {
                if !c.load(Ordering::Relaxed) {
                    if let Ok(mut state) = s.lock() { state.error = Some(error); }
                }
            }
            if let Ok(mut slot) = p.lock() {
                if let Some(mut child) = slot.take() { let _ = child.kill(); let _ = child.wait(); }
            }
        });
        Self { input, state, cancelled, child }
    }
}
impl Drop for LiveJob {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Relaxed);
        if let Ok(mut slot) = self.child.lock() { if let Some(child) = slot.as_mut() { let _ = child.kill(); } }
    }
}

fn header(reader: &mut impl BufRead) -> Result<serde_json::Value, String> {
    let mut text = String::new();
    reader.take(65537).read_line(&mut text).map_err(|e| e.to_string())?;
    if text.len() > 65536 || text.is_empty() { return Err("实时拆轨进程已停止，详情见 backend/live-last-job.log".into()); }
    serde_json::from_str(&text).map_err(|e| format!("实时协议错误：{e}"))
}
fn number(v: &serde_json::Value, key: &str) -> Result<u64, String> {
    v[key].as_u64().ok_or_else(|| format!("实时结果缺少 {key}"))
}
fn run(rate: u32, model: SeparationModel, input: &Arc<Mutex<LiveInput>>, state: &Arc<Mutex<LiveState>>,
    cancel: &AtomicBool, child_slot: &Arc<Mutex<Option<Child>>>) -> Result<(), String> {
    if !(8000..=192000).contains(&rate) { return Err("实时模式支持 8–192 kHz 音源".into()); }
    let base = separation::root();
    let bundled = base.join("backend/runtime/python.exe");
    let python = if bundled.is_file() { bundled } else { base.join("backend/.venv/Scripts/python.exe") };
    let log = std::fs::File::create(base.join("backend/live-last-job.log")).map_err(|e| e.to_string())?;
    let mut command = Command::new(python);
    command.args(["-I", "-u"]).arg(base.join("backend/live.py")).args(["--model", model.filename(), "--rate", &rate.to_string()])
        .current_dir(&base).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(log);
    #[cfg(windows)] { use std::os::windows::process::CommandExt; command.creation_flags(0x08000000); }
    if cancel.load(Ordering::Relaxed) { return Ok(()); }
    let mut child = command.spawn().map_err(|e| format!("无法启动实时拆轨：{e}"))?;
    let mut writer = child.stdin.take().ok_or("实时输入不可用")?;
    let mut reader = BufReader::new(child.stdout.take().ok_or("实时输出不可用")?);
    *child_slot.lock().map_err(|_| "实时任务锁异常")? = Some(child);
    if cancel.load(Ordering::Relaxed) { return Ok(()); }
    let names = loop {
        let value = header(&mut reader)?;
        match value["type"].as_str() {
            Some("status") => { if let Ok(mut s) = state.lock() { s.message = value["text"].as_str().unwrap_or("加载模型…").into(); } }
            Some("error") => return Err(value["text"].as_str().unwrap_or("实时模型加载失败").into()),
            Some("ready") => {
                let names: Vec<String> = value["names"].as_array().ok_or("实时轨道清单错误")?.iter()
                    .map(|v| v.as_str().unwrap_or("未知轨道").to_string()).collect();
                if !(2..=7).contains(&names.len()) || number(&value, "rate")? != OUTPUT_RATE as u64
                    || number(&value, "pre_roll")? != PRE_ROLL as u64 || number(&value, "frames")? != RESULT_FRAMES as u64 { return Err("实时轨道格式错误".into()); }
                if let Ok(mut s) = state.lock() { s.names = names.clone(); s.message = "等待积攒 6 秒声音…".into(); s.reserved_mb = value["reserved_mb"].as_u64().unwrap_or(0); }
                break names;
            }
            _ => return Err("实时启动响应错误".into()),
        }
    };
    let mut last = None;
    while !cancel.load(Ordering::Relaxed) {
        let window = input.lock().ok().and_then(|s| s.latest(last));
        let Some(window) = window else { std::thread::sleep(Duration::from_millis(20)); continue; };
        let skipped = last.map(|last| (window.end - last).saturating_sub(rate as u64 * HOP as u64) / (rate as u64 * HOP as u64)).unwrap_or(0);
        let request = serde_json::json!({"seq": window.end, "rate": rate, "frames": window.frames.len()});
        writeln!(writer, "{request}").map_err(|e| e.to_string())?;
        let mut bytes = Vec::with_capacity(window.frames.len() * 8);
        for frame in &window.frames { for value in frame { bytes.extend_from_slice(&value.to_le_bytes()); } }
        writer.write_all(&bytes).and_then(|_| writer.flush()).map_err(|e| e.to_string())?;
        let value = header(&mut reader)?;
        if value["type"] == "error" { return Err(value["text"].as_str().unwrap_or("实时推理失败").into()); }
        if value["type"] != "result" || number(&value, "seq")? != window.end || number(&value, "tracks")? != names.len() as u64
            || number(&value, "frames")? != RESULT_FRAMES as u64 || number(&value, "pre_roll")? != PRE_ROLL as u64
            || number(&value, "rate")? != OUTPUT_RATE as u64 { return Err("实时结果尺寸或顺序不一致".into()); }
        let frames = RESULT_FRAMES;
        let mut bytes = vec![0u8; frames * names.len() * 4];
        reader.read_exact(&mut bytes).map_err(|e| e.to_string())?;
        let tracks = bytes.chunks_exact(frames * 4).map(|track| track.chunks_exact(4).map(|f|
            [i16::from_le_bytes([f[0], f[1]]), i16::from_le_bytes([f[2], f[3]])]).collect()).collect();
        let song = Arc::new(Song { tracks, names: names.clone(), rate: OUTPUT_RATE as u32, name: "实时分轨".into() });
        if let Ok(mut s) = state.lock() {
            s.skipped += skipped + u64::from(s.latest.is_some());
            s.latest = Some(LiveBlock { song, source_start: window.captured - Duration::from_secs(4) });
            s.inference_secs = value["seconds"].as_f64().unwrap_or(0.0) as f32;
            s.reserved_mb = value["reserved_mb"].as_u64().unwrap_or(0);
            s.message = "实时分轨运行中".into();
        }
        last = Some(window.end);
    }
    Ok(())
}

pub struct LiveDisplay {
    block: LiveBlock,
    comparison: SpectralComparison,
    began: Instant,
}
impl LiveDisplay {
    pub fn new(block: LiveBlock) -> Self { Self { comparison: SpectralComparison::new(block.song.tracks.len()), block, began: Instant::now() } }
    pub fn finished(&self) -> bool { self.began.elapsed().as_secs_f64() >= HOP as f64 }
    pub fn latency(&self) -> f32 {
        let position = self.began.elapsed().min(Duration::from_secs(HOP as u64));
        Instant::now().saturating_duration_since(self.block.source_start + position).as_secs_f32()
    }
    pub fn view(&mut self, enabled: &[bool]) -> ComparisonView {
        if enabled.len() == self.comparison.enabled.len() { self.comparison.enabled.copy_from_slice(enabled); }
        let end = (self.began.elapsed().as_secs_f64() * OUTPUT_RATE as f64).min((OUTPUT_RATE * HOP) as f64) + PRE_ROLL as f64;
        self.comparison.update(&self.block.song, end);
        self.comparison.view(OUTPUT_RATE as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn live_input_retains_only_latest_window_and_requires_new_audio() {
        let mut input = LiveInput::new(10);
        input.push(&vec![[0.0, 0.0]; 59]);
        assert!(input.latest(None).is_none());
        input.push(&[[1.0, 1.0]]);
        let first = input.latest(None).unwrap();
        assert_eq!(first.end, 60);
        input.push(&vec![[2.0, 2.0]; 100]);
        let next = input.latest(Some(first.end)).unwrap();
        assert_eq!(next.frames.len(), 60);
        assert!(next.frames.iter().all(|f| f[0] == 2.0));
        assert!(input.latest(Some(next.end)).is_none());
        input.push(&vec![[3.0, 3.0]; 19]);
        assert!(input.latest(Some(next.end)).is_none());
        input.push(&[[3.0, 3.0]]);
        assert!(input.latest(Some(next.end)).is_some());
    }
    #[test]
    fn protocol_rejects_eof_oversized_and_malformed_headers() {
        assert!(header(&mut std::io::Cursor::new(Vec::<u8>::new())).is_err());
        assert!(header(&mut std::io::Cursor::new(vec![b' '; 70000])).is_err());
        assert!(header(&mut std::io::Cursor::new(b"invalid\n")).is_err());
        assert_eq!(header(&mut std::io::Cursor::new(b"{\"rate\":44100}\n")).unwrap()["rate"], 44100);
    }
    #[test]
    fn capture_gap_discards_old_context_and_waits_for_a_fresh_window() {
        let mut input = LiveInput::new(10);
        input.push(&[[1.0, 1.0]; 60]);
        let previous = input.latest(None).unwrap().end;
        input.captured = Instant::now() - Duration::from_secs(2);
        input.push(&[[2.0, 2.0]; 20]);
        assert_eq!(input.frames.len(), 20);
        assert!(input.latest(Some(previous)).is_none());
        input.push(&[[2.0, 2.0]; 40]);
        let fresh = input.latest(Some(previous)).unwrap();
        assert!(fresh.frames.iter().all(|f| f[0] == 2.0));
        assert_eq!(fresh.end, 120);
    }
    #[test]
    #[ignore = "requires installed local models and NVIDIA GPU"]
    fn persistent_worker_delivers_real_stems_and_cancels_during_running_and_loading() {
        let job = LiveJob::start(48000, SeparationModel::Six);
        let slot = job.child.clone();
        let frames: Vec<[f32; 2]> = (0..48000 * 6).map(|i| {
            let v = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin() * 0.12;
            [v, v * 0.8]
        }).collect();
        job.input.lock().unwrap().push(&frames);
        let began = Instant::now();
        let block = loop {
            {
                let mut state = job.state.lock().unwrap();
                assert!(state.error.is_none(), "{:?}", state.error);
                if let Some(block) = state.latest.take() {
                    eprintln!("live inference {:.3}s; reserved {} MB", state.inference_secs, state.reserved_mb);
                    break block;
                }
            }
            assert!(began.elapsed() < Duration::from_secs(90), "live worker timed out");
            std::thread::sleep(Duration::from_millis(50));
        };
        assert_eq!(block.song.tracks.len(), 7);
        assert!(block.song.names.iter().any(|s| s == "吉他"));
        assert!(block.song.tracks.iter().all(|s| s.len() == RESULT_FRAMES));
        assert!(block.song.tracks[1..].iter().any(|s| s.iter().any(|f| f[0] != 0)));
        let mut display = LiveDisplay::new(block);
        // A new chunk must render real frequency content immediately, without a zero-prefix dip.
        let view = display.view(&[false, true, true, true, true, true, true]);
        assert_eq!(view.layers.len(), 6);
        assert!(view.layers.iter().any(|l| l.spectrum_db.iter().any(|db| *db > -90.0)));
        drop(job);
        let deadline = Instant::now();
        while slot.lock().unwrap().is_some() {
            assert!(deadline.elapsed() < Duration::from_secs(10), "running child was not reaped");
            std::thread::sleep(Duration::from_millis(20));
        }
        let loading = LiveJob::start(48000, SeparationModel::Four);
        let slot = loading.child.clone();
        let deadline = Instant::now();
        while slot.lock().unwrap().is_none() {
            assert!(deadline.elapsed() < Duration::from_secs(5), "child did not start");
            std::thread::sleep(Duration::from_millis(10));
        }
        drop(loading);
        let deadline = Instant::now();
        while slot.lock().unwrap().is_some() {
            assert!(deadline.elapsed() < Duration::from_secs(10), "loading child was not reaped");
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}
