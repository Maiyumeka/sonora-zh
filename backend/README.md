# 本地拆轨后端

此目录只包含桥接源码和依赖版本清单，不包含 Python、安装包、CUDA、FFmpeg、模型、歌曲或缓存。基础实时频谱功能不依赖此后端。

## 自行准备环境（Windows x64）

先安装 Python 3.12 x64。在仓库根目录执行：

```powershell
py -3.12 -m venv backend/.venv
backend/.venv/Scripts/python.exe -m pip install -r backend/requirements.lock.txt
```

`requirements.lock.txt` 为原开发机导出的精确版本清单，包含 PyTorch 2.8.0+cu128，并使用 PyTorch 官方 CUDA 12.8 轮子源。这会另行下载较大的依赖；安装结果受 Python 版本、软件源及平台支持影响，本仓库不提供这些安装包，也不保证所有未来软件源始终保留相同版本。GPU 运行还需合适的 NVIDIA 驱动。CPU 可用于离线拆轨，实时模式要求兼容的 NVIDIA GPU。

可自行准备 `backend/bin/ffmpeg.exe`。若缺少此文件，离线脚本会尝试从已安装的 imageio-ffmpeg 中复制可执行文件。FFmpeg 的具体许可证取决于所选构建；本仓库不分发它。

使用模型前请阅读 [第三方与模型说明](../THIRD_PARTY.md)。离线脚本的 `audio-separator` 加载流程在本地缺少模型时可能自动下载到 `backend/models/`，首次需要联网；实时脚本只读取这个目录里的现有模型描述和权重。模型许可需单独核实。

程序优先查找 `backend/runtime/python.exe`，随后查找 `backend/.venv/Scripts/python.exe`。源码安装通常使用后者。移动 venv 容易使其基础 Python 路径失效，应在目标机器重新创建。

## 运行方式

- `separate.py`：每首歌曲启动一个进程，完成后退出。按输入内容哈希及模型名缓存，`ready.json` 在所有对齐的 PCM16 立体声轨道就绪后原子写入。
- `live.py`：常驻 GPU 进程，只用于实时曲线，不播放或保存捕获音频。读取最近 6 秒，以 2 秒为步长处理，显示中央 2 秒并附加 4096 帧 FFT 前导。
- 两种模式均使用 2 个 CPU 计算线程，PyTorch CUDA 分配器上限为显存的 50%；这个上限不包括所有驱动开销。离线每首最长 10 分钟，实时模式默认关闭。
- 试听在 Rust 中混合所选轨道；曲线选择独立于试听选择，所有曲线使用同一时间位置和绝对电平刻度。

脚本不把歌曲上传到服务器。首次依赖安装及模型下载需要网络，之后可使用本地资源。`backend/songs/` 保存离线解码和分轨 WAV，`cache/` 与 `tmp/` 保存临时状态，错误日志为 `last-job.log`、`live-last-job.log`。这些目录、运行环境和日志均被 Git 忽略。

## 协议与测试

离线 stdout 为 UTF-8 行协议：`SONORA_STATUS`、`SONORA_ERROR`、`SONORA_METRICS`、`SONORA_DONE`。实时使用换行分隔 JSON 头及小端二进制负载：输入为立体声 float32，输出为 44.1 kHz 立体声 PCM16；验证请求序号、长度与轨道数。实时队列有上限，停顿或切换音源会丢弃旧上下文。

仓库根目录的 `cargo test --release --locked` 默认不运行依赖 GPU 和本地模型的集成测试。仅在自行配置好环境和模型后，才运行：

```powershell
cargo test --release --locked persistent_worker_delivers_real_stems -- --ignored --nocapture
```
