# Sonora 中文改版

基于 [Px228-Da-Da/sonora](https://github.com/Px228-Da-Da/sonora) 的非官方修改版：在 Windows 上观察正在播放的声音，并可通过可选的本地 Python 后端拆轨、试听和比较乐器频谱。

## 下载成品（Windows x64）

到 [Releases](https://github.com/Maiyumeka/sonora-zh/releases/latest) 下载 `Sonora-ZH-Windows-x64-*.zip`，完整解压后双击 `sonora-zh.exe`。提供 SHA256 校验文件；程序未进行数字签名。

这是免安装轻量版，实时音频分析可以直接使用；**拆轨、多轨试听和分轨曲线仍需自行准备 Python 后端及模型**，详见 [后端说明](backend/README.md)。成品不附带模型、Python/CUDA/FFmpeg 运行环境、歌曲、缓存或日志；如系统提示缺少 C++ 运行库，请从微软官方安装 x64 版本。

## 改动

- 中文界面与字体加载，四种仪表整合到可调整大小的普通窗口。
- 系统声音、输入设备与 Windows 指定应用音频捕获。
- 可选的四轨/六轨离线拆分，按实际输出生成轨道选择；六轨增加吉他、钢琴。
- 多轨同步试听、暂停/跳转、试听音量和限峰；可用不同颜色同时比较多条频谱。
- 声谱图亮度范围调节与默认关闭的音量补偿。
- 实验性实时拆轨曲线开关，默认关闭，只显示曲线，原播放器继续播放。
- 限制推理线程、片段长度和队列大小，避免持续积压。

六轨不保证每种乐器完全独立，钢琴分离为实验性质。实时曲线通常落后声音约 4 秒加推理时间，需要兼容的 NVIDIA 显卡；它不是零延迟的实时分轨播放器。

## 构建与使用

目前面向 Windows x64。安装支持 Rust 2024 edition 的 Rust 工具链及 MSVC C++ 构建工具；原开发环境使用 Rust 1.98.1。

```powershell
git clone https://github.com/Maiyumeka/sonora-zh.git
cd sonora-zh
cargo build --release --locked
cargo test --release --locked
cargo run --release --locked
```

程序生成在 `target/release/sonora.exe`。打开后选择系统输出设备并在其他播放器中播放音乐，即可看实时仪表，无需外置声卡。

需要拆轨时，按 [后端安装说明](backend/README.md) 自行准备环境与模型。开发时 `cargo run` 会回退查找源码目录的 `backend`；移动可执行文件时须把 `backend` 放在它旁边。

详细操作见 [中文使用说明](docs/使用说明.md)。[上游原始 README](docs/upstream-README.md) 留作来源记录，安装和界面操作以本改版说明为准。

## 许可与来源

原项目由 Px228-Da-Da 发布，采用 MIT 许可。根目录 [LICENSE](LICENSE) 完整保留上游的 `Copyright (c) 2026 Px228-Da-Da` 与许可正文。本仓库新增修改也采用 MIT 许可，来源和修改范围见 [NOTICE.md](NOTICE.md)。本项目不代表上游官方版本。

第三方依赖适用各自许可，模型权重的授权也与代码分开；本仓库不提供权重，也不声明权重属于 MIT。见 [THIRD_PARTY.md](THIRD_PARTY.md)。
