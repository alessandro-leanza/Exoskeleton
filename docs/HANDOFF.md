# Handoff

## Objective and current state

Make the GitHub entry page clearer for robotics/AI recruiters and research
reviewers. First pass focuses on the English README and supporting documentation.
Alessandro approved commit and push on 2026-10-22. Delivery branch: `main`.
The draft was prepared on `docs/repository-frontpage`.

The checkout was fast-forwarded from `ccac469` to `3838d6e` (2026-10-21) before
editing, preserving the upstream figures and research overview.

## Decisions and changes

- README now opens with the project purpose, stack, laboratory photo, and clear
  navigation, followed by capabilities, architecture, qualified results, code
  entry points, and separate offline/hardware routes.
- `docs/RESEARCH.md` preserves all three original result rows verbatim, including
  sample sizes and comparison conditions, and documents implementation scope.
- `docs/SETUP.md` holds the existing launch workflow and legacy ODrive notes.
  Corrected Conda wording: the local `ros2_env/pyvenv.cfg` identifies a Python
  3.10 virtual environment. No installation or hardware execution was attempted.
- No runtime code, dependencies, model files, or experiment data were changed.
- Moved the offline guide to `docs/OFFLINE_BENCHMARK.md` and removed internal
  supervisor/student wording. Replaced outdated perception instructions with
  links to the maintained guides. Development diff/stat files are in
  `docs/archive/` rather than the repository root.
- Added a minimal Python `.gitignore` and removed 37 tracked `.pyc` files from
  the index. All local cache files, including existing modified ones, are kept.

## Verified

- Markdown parsed; 40 relative links/anchors resolved; both images have alt text.
- Eleven Bash blocks pass `bash -n`; `git diff --check` passes.
- Original research result rows remain unchanged in `docs/RESEARCH.md`.
- Desktop/mobile screenshots of a local HTML rendering were visually inspected.
  Preview: `/tmp/exoskeleton-readme-preview.html` (approximation of GitHub styling).
- Code inspection: local Qwen inference runs synchronously before the live gate
  update; VLM estimates do not update controller mass `m_b`. The separate remote
  estimate is asynchronous but only stored/printed. Removed the overview's
  unsupported implication of an independently scheduled local VLM control path.

## Constraints and next steps

- Reported study results were retained, not recomputed; the linked preprint is
  earlier than the overview and does not establish all its later results.
- SHA-256 checks confirm all 37 previously tracked cache files retain their
  local contents. Cache paths are now ignored. The untracked local detector
  `best-tobii-3objs-v2.pt` remains outside the commit; preserve it.
- No root license file exists; package metadata contains Apache/MIT/TODO license
  declarations. Confirm intended licensing/ownership before adding a license.
- Fresh-machine reproducibility remains incomplete: dependency versions are not
  pinned; the live detector's default checkpoint is not tracked; paths are
  laboratory-specific.
- Next: confirm the intended license and improve fresh-machine reproducibility
  in a separate change. This README balances portfolio and scientific review;
  no specific audience preference was supplied.
