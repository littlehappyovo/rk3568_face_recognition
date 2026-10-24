# 本地播报音频

这5段音频由sherpa-onnx 2.0.0使用`vits-melo-tts-zh_en`生成，统一转换为48kHz、双声道、16位PCM WAV。板端播放WAV，不安装TTS模型或Python依赖。语音段以RMS0.14为目标、峰值0.8为上限归一化，前后添加200ms/100ms静音以等待音频通路稳定；实际响度还取决于系统音量、DAC、功放和扬声器。

| 文件 | 内容 |
| --- | --- |
| morning.wav | 早上好。 |
| noon.wav | 中午好。 |
| afternoon.wav | 下午好。 |
| evening.wav | 晚上好。 |
| denied.wav | 无权限，请联系管理员。 |

`manifest.json`记录模型哈希、音频哈希和时长，模型许可见[MyShell许可](../third_party_licenses/MeloTTS_LICENSE.txt)。模型不随仓库分发。来源与使用方式见[sherpa-onnx官方模型说明](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html#vits-melo-tts-zh-en-chinese-english-1-speaker)。

需要重新生成时，在开发PC的独立虚拟环境中安装`sherpa-onnx==2.0.0`和`numpy`，从官方说明下载并解压`vits-melo-tts-zh_en.tar.bz2`，然后执行：

```bash
python scripts/generate_voice_assets.py --model-dir /path/to/vits-melo-tts-zh_en
```

生成结果受模型噪声采样影响，不承诺每次生成的文件哈希一致。替换音频需保留这5个文件名并使用有效PCM WAV，重新构建或复制到可执行文件旁的`audio/`。

板端扬声器测试：

```bash
bash scripts/setup_rk809_audio.sh
aplay -q -D plughw:CARD=rockchiprk809,DEV=0 audio/morning.wav
```

默认DAC音量90%，可用`FACEAPP_AUDIO_VOLUME=80%`降低。MIPI启动脚本会在关闭GDM后设置RK809通路；直接从桌面运行时先执行该设置脚本。播放设备和问候时段见`config/voice.json`。本配置使用板载扬声器通路；外接耳机需要按实际硬件更改通路。
