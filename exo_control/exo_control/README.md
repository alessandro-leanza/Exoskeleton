# Perception pipeline

The perception code combines YOLO detections, region-of-interest cropping, and
Qwen2.5-VL inference. It supports offline image/video evaluation and is also
integrated into the Tobii application.

- [Offline benchmark: inputs, configuration, and CSV outputs](../../docs/OFFLINE_BENCHMARK.md)
- [Live laboratory setup and launch commands](../../docs/SETUP.md)
- [Research results and current implementation scope](../../docs/RESEARCH.md)
- [Project overview](../../README.md)

The offline entry point is [test_pipeline.py](test_pipeline.py); configure its
input paths and detector weights before running it. Benchmark outputs are
written to timestamped files under `results/`.

The pipeline requires `grasped` detections and applies confidence and stability
gates. Keep the evaluation protocol fixed when comparing runs; see the benchmark
guide for image/video stability settings and ground-truth matching.
