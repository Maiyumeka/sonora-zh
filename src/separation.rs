//! One child process per job; cancelling or closing the app stops inference.
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use crate::player::Song;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SeparationModel { Four, Six }
impl SeparationModel {
    pub fn filename(self) -> &'static str {
        match self { Self::Four => "htdemucs.yaml", Self::Six => "htdemucs_6s.yaml" }
    }
    pub fn label(self) -> &'static str {
        match self { Self::Four => "四轨（快速）", Self::Six => "六轨（增加吉他、钢琴）" }
    }
}

pub struct JobState {
    pub message: String,
    pub result: Option<Result<Arc<Song>, String>>,
    pub finished: bool,
}
pub struct Job {
    pub state: Arc<Mutex<JobState>>,
    child: Arc<Mutex<Option<Child>>>,
    cancelled: Arc<AtomicBool>,
}

pub fn root() -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let parent = exe.parent().unwrap_or(Path::new(".")).to_path_buf();
    if parent.join("backend/separate.py").is_file() { parent } else { PathBuf::from(env!("CARGO_MANIFEST_DIR")) }
}

impl Job {
    pub fn start(source: PathBuf, cpu: bool, model: SeparationModel) -> Self {
        let state = Arc::new(Mutex::new(JobState { message: "启动本地拆轨…".into(), result: None, finished: false }));
        let child = Arc::new(Mutex::new(None));
        let cancelled = Arc::new(AtomicBool::new(false));
        let (s, c, cancel) = (state.clone(), child.clone(), cancelled.clone());
        std::thread::spawn(move || {
            let result = run(&source, cpu, model, &s, &c, &cancel);
            if let Ok(mut state) = s.lock() {
                state.message = if cancel.load(Ordering::Relaxed) { "已取消拆轨".into() } else if result.is_ok() { "分轨已就绪".into() } else { "拆轨未完成".into() };
                state.result = if cancel.load(Ordering::Relaxed) { None } else { Some(result) };
                state.finished = true;
            }
        });
        Self { state, child, cancelled }
    }
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
        if let Ok(mut slot) = self.child.lock() {
            if let Some(child) = slot.as_mut() { let _ = child.kill(); }
        }
    }
}
impl Drop for Job { fn drop(&mut self) { self.cancel(); } }

fn run(source: &Path, cpu: bool, model: SeparationModel, state: &Arc<Mutex<JobState>>, slot: &Arc<Mutex<Option<Child>>>, cancelled: &AtomicBool) -> Result<Arc<Song>, String> {
    let base = root();
    let bundled_python = base.join("backend/runtime/python.exe");
    let python = if bundled_python.is_file() { bundled_python } else { base.join("backend/.venv/Scripts/python.exe") };
    if !python.is_file() { return Err("拆轨环境未安装，请检查 backend 文件夹".into()); }
    let log = std::fs::File::create(base.join("backend/last-job.log")).map_err(|e| e.to_string())?;
    let mut command = Command::new(python);
    command.arg("-I").arg("-u").arg(base.join("backend/separate.py")).arg(source)
        .args(["--device", if cpu { "cpu" } else { "auto" }])
        .args(["--model", model.filename()])
        .env("PYTHONIOENCODING", "utf-8").env("OMP_NUM_THREADS", "2").env("MKL_NUM_THREADS", "2")
        .current_dir(&base).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::from(log));
    #[cfg(windows)] {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    let mut child = command.spawn().map_err(|e| format!("无法启动拆轨：{e}"))?;
    let stdout = child.stdout.take().ok_or("无法读取拆轨进度")?;
    {
        let mut guard = slot.lock().map_err(|_| "任务状态异常")?;
        if cancelled.load(Ordering::Relaxed) { let _ = child.kill(); }
        *guard = Some(child);
    }
    let mut directory = None;
    let mut error = None;
    for line in BufReader::new(stdout).lines() {
        let line = line.map_err(|e| e.to_string())?;
        if let Some(message) = line.strip_prefix("SONORA_STATUS ") {
            if let Ok(mut s) = state.lock() { s.message = message.into(); }
        } else if let Some(path) = line.strip_prefix("SONORA_DONE ") {
            directory = Some(PathBuf::from(path));
        } else if let Some(message) = line.strip_prefix("SONORA_ERROR ") { error = Some(message.to_string()); }
    }
    let exit = slot.lock().map_err(|_| "任务状态异常")?.as_mut().ok_or("任务状态异常")?.wait().map_err(|e| e.to_string())?;
    if cancelled.load(Ordering::Relaxed) { return Err("已取消".into()); }
    if !exit.success() { return Err(error.unwrap_or("拆轨进程未完成，详情见 backend/last-job.log".into())); }
    let dir = directory.ok_or("未找到完整的分轨结果")?;
    if let Ok(mut s) = state.lock() { s.message = "载入分轨音频…".into(); }
    let name = source.file_name().unwrap_or_default().to_string_lossy().into_owned();
    Ok(Arc::new(Song::load(&dir, name)?))
}

/// Native Windows audio-file picker, without another UI runtime dependency.
pub fn pick_file() -> Option<PathBuf> {
    use windows::core::{PCWSTR, PWSTR};
    use windows::Win32::UI::Controls::Dialogs::{GetOpenFileNameW, OPENFILENAMEW, OFN_EXPLORER, OFN_FILEMUSTEXIST, OFN_NOCHANGEDIR};
    let mut file = vec![0u16; 32768];
    let filter: Vec<u16> = "音频文件\0*.wav;*.mp3;*.flac;*.ogg;*.opus;*.m4a;*.aac;*.aiff\0所有文件\0*.*\0\0".encode_utf16().collect();
    let title: Vec<u16> = "选择要拆轨的歌曲\0".encode_utf16().collect();
    let mut dialog = OPENFILENAMEW {
        lStructSize: std::mem::size_of::<OPENFILENAMEW>() as u32,
        lpstrFilter: PCWSTR(filter.as_ptr()), lpstrFile: PWSTR(file.as_mut_ptr()), nMaxFile: file.len() as u32,
        lpstrTitle: PCWSTR(title.as_ptr()), Flags: OFN_FILEMUSTEXIST | OFN_EXPLORER | OFN_NOCHANGEDIR,
        ..Default::default()
    };
    if !unsafe { GetOpenFileNameW(&mut dialog) }.as_bool() { return None; }
    let end = file.iter().position(|&x| x == 0)?;
    Some(PathBuf::from(String::from_utf16_lossy(&file[..end])))
}

#[cfg(test)]
mod model_tests {
    use super::*;
    #[test]
    fn model_options_select_different_backend_models() {
        assert_eq!(SeparationModel::Four.filename(), "htdemucs.yaml");
        assert_eq!(SeparationModel::Six.filename(), "htdemucs_6s.yaml");
        assert_ne!(SeparationModel::Four.filename(), SeparationModel::Six.filename());
    }
}
