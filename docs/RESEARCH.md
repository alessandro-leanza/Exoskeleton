# Research results and implementation scope

[Back to the project overview](../README.md)

## Experimental results

The evaluations below concern controlled laboratory stooping and lifting tasks.
These measurements are retained from the repository's research overview; this
documentation update does not recompute them. The full manuscript PDF is not
included in the repository. The [arXiv preprint](https://arxiv.org/abs/2602.04648)
is an earlier version and may not contain all figures or experimental results
summarized here.

| Evaluation | Selected finding | Sample and comparison |
| --- | --- | --- |
| Gaze ablation | Mean payload-aware assistance-onset delay: **0.066 ± 0.033 s** with gaze versus **1.892 ± 0.040 s** without gaze; mean anticipation **1.826 s** | One participant, 50 randomized trials per condition. Onset is referenced to the first YOLO detection of the object as grasped, not a direct measurement of physical load transfer. |
| Questionnaire | Physical-demand Q1 score: **3.60 / 4** with vision gating, **2.13 / 4** with posture-only assistance, **2.00 / 4** without the exoskeleton. Higher means less perceived physical demand. | 15 participants; five repetitions per condition. These ratings evaluate the integrated assistance strategy. |
| Preliminary EMG | Loaded-condition envelope RMS reduction: **14.29%** for thoracic and **14.63%** for lumbar erector spinae | Five participants; complete vision-enabled exoskeleton versus no exoskeleton during loaded lifting. |

### Interpretation limits

- The gaze ablation measures detection-referenced onset in one participant;
  it does not measure physical load-transfer timing directly.
- The preliminary EMG study did not include a posture-only assistance condition,
  so these muscular-activation reductions cannot be independently attributed to
  gaze or vision gating.
- The current experiments do not independently separate task-phase anticipation
  from payload-dependent torque adaptation.
- These findings do not establish performance across arbitrary users, objects,
  or environments. The detector weights were developed for the laboratory setup.

## Implementation scope

The architecture figure describes the research framework. The current source
contains both the live control stack and experimental perception code; it is
not a frozen reproduction package for every reported experiment.

| Component | Current source behavior |
| --- | --- |
| Live grasp and gaze gate | [`tobii_vlm/__main__.py`](../exo_control/exo_control/tobii_vlm/__main__.py) detects grasp state and publishes `perception/box_gate`. |
| Assistance | [`admittancecontrol_box.py`](../exo_control/exo_control/admittancecontrol_box.py) combines joint feedback and the gate with a finite-state policy. Payload mass is configured through the ROS parameter `m_b`. |
| Local VLM | [`pipeline/`](../exo_control/exo_control/pipeline/) estimates object name, weight category, and fragility category. Results are logged; no connection from these outputs to the controller's payload parameter is implemented. |
| Live inference scheduling | YOLO runs in a worker thread. The local Qwen pipeline is called synchronously inside the live perception loop, before the gate update; its latency can delay that update. This snapshot does not provide an independently scheduled, nonblocking local VLM path. |
| Optional remote weight estimate | The live app has a separate asynchronous request enabled by `OPENAI_API_KEY`. The parsed estimate is stored and printed; it does not update `m_b`. |
| Offline evaluation | [`test_pipeline.py`](../exo_control/exo_control/test_pipeline.py) evaluates image or video inputs without ROS or motors and writes timestamped CSVs. It does not reproduce the human-subject experiments above. |

The offline harness uses keyword matching against the VLM's predicted object
name to select configured ground truth. Treat its accuracy summaries as
exploratory diagnostics; inspect the per-frame CSVs and unmatched predictions
when interpreting comparisons. See the [benchmark guide](OFFLINE_BENCHMARK.md)
for the existing protocol.

## Figures

The [architecture](../images/architecture.png) and
[experimental setup](../images/setup.webp) figures are reused from the supplied
research manuscript. Source figures, dimensions, and synchronization notes are
recorded in [images/README.md](../images/README.md).
