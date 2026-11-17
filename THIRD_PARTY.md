# 第三方依赖与模型

这里记录代码依赖与授权边界。本仓库不内置这些第三方库的源码或二进制，Rust 依赖由 Cargo 获取，Python 依赖需使用者另行安装。依赖版本分别记录在 Cargo.lock 和 backend/requirements.lock.txt 中；这些版本清单不是运行环境本体。

| 项目 | 用途 | 许可与来源 |
| --- | --- | --- |
| [Sonora](https://github.com/Px228-Da-Da/sonora) | 本项目的上游源码 | [MIT](https://github.com/Px228-Da-Da/sonora/blob/v1.0.0/LICENSE)，本仓库完整保留于 LICENSE |
| [egui / eframe](https://github.com/emilk/egui) | 窗口与界面 | MIT OR Apache-2.0，见其项目许可 |
| [CPAL](https://github.com/RustAudio/cpal) | 音频捕获和播放 | Apache-2.0，见其项目许可 |
| [RustFFT](https://github.com/ejmahler/RustFFT) | FFT | MIT OR Apache-2.0，见其项目许可 |
| [hound](https://github.com/ruuda/hound) | WAV 读取 | Apache-2.0，见其项目许可 |
| [serde_json](https://github.com/serde-rs/json) | 协议与清单 | MIT OR Apache-2.0，见其项目许可 |
| [windows-rs](https://github.com/microsoft/windows-rs) | Windows API | MIT OR Apache-2.0，见其项目许可 |
| [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) | 离线分离和所含 Demucs 实现的接口 | [MIT](https://github.com/nomadkaraoke/python-audio-separator/blob/v0.48.0/LICENSE)，其依赖仍适用各自许可 |
| [Demucs](https://github.com/facebookresearch/demucs) | 分离算法 | [代码 MIT](https://github.com/facebookresearch/demucs/blob/main/LICENSE)，不据此推断预训练权重许可 |
| [PyTorch](https://github.com/pytorch/pytorch) | 推理 | 以其 LICENSE 和第三方声明为准 |
| [FFmpeg](https://ffmpeg.org/legal.html) | 音频解码 | 随具体构建选项适用 LGPL/GPL 等，未在此分发 |

## 预训练模型不包含在本仓库

代码支持 `htdemucs` 和 `htdemucs_6s`，本仓库不包含 YAML 模型描述、权重或模型缓存。**代码的 MIT 许可不自动覆盖预训练权重。**

Demucs 维护者曾在 [2022 年的许可讨论](https://github.com/facebookresearch/demucs/issues/327#issuecomment-1134828611) 中明确区分代码和模型权重，并限定当时权重的科学研究用途。这份早于 v4 的说明不能作为当前所有权重的明确授权；使用者应核实所选模型的具体条款，尤其在商业使用或再次分发前。本仓库不对模型另行授予许可。

离线后端沿用 audio-separator 的模型加载流程，本地缺少模型时可能访问第三方模型下载服务；实时后端只读取已经准备好的本地模型。请在运行拆轨功能前确认模型来源及适用授权。

## 如需再分发二进制或完整环境

需要重新审核所有实际打包依赖的许可、版权声明及对应源码义务；FFmpeg 构建、CUDA 运行库、Python 与 MSVC 运行库须分别处理。根目录 MIT 许可不替代上述要求。本源码仓库不附带已完成完整运行包分发合规审核的声明。
