# Third-party notices

This repository distributes the application source, configuration examples and regression tests. Model binaries, the Rockchip SDK/runtime, recorded face images and generated identity galleries are separate local inputs.

- YuNet and SFace model source revisions and expected checksums are recorded in `models/MODELS.json`. Their upstream license texts are preserved in `third_party_licenses/`.
- Obtain `rknn_api.h` and `librknnrt.so` from the Rockchip SDK/BSP matching the target device. These vendor files are not redistributed here; see `third_party/rknn/README.md`.
- Qt, OpenCV and compiler/runtime dependencies are supplied by the target system.
- The five fixed speech prompts are generated using sherpa-onnx 2.0.0 and the MyShell MeloTTS Chinese model. Model and audio hashes are recorded in `audio/manifest.json`; the model license is preserved in `third_party_licenses/MeloTTS_LICENSE.txt`. The synthesis model is not redistributed.
