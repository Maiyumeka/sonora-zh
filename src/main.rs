//! Sonora — real-time audio metering for the desktop, written in Rust.
//! Captures system audio (WASAPI loopback) or any input device and shows a
//! spectrogram, spectrum analyzer, LUFS/peak loudness and a stereo goniometer.
//! Inspired by MiniMeters (https://minimeters.app).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod analysis;
mod audio;
#[cfg(windows)]
mod proc_audio;
mod ui;
mod player;
mod comparison;
mod separation;
mod live;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 780.0])
            .with_min_inner_size([840.0, 620.0])
            .with_title("Sonora 中文版"),
        ..Default::default()
    };

    eframe::run_native(
        "Sonora 中文版",
        options,
        Box::new(|cc| Ok(Box::new(ui::MetersApp::new(cc)))),
    )
}
