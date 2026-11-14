# Changelog

## 0.2.3 (2026-11-14)

- The download of the weights is the core's `xblick.weights` (xblick 0.5.3). The node's own copy of that code is
  gone. `weights.py` keeps the node's files, its terms notice and its `STORE`. The options, the messages, and the
  files and their places stay as they were.
- The node needs `xblick>=0.5.3,<0.6`.
- `tests/test_weights.py` keeps the node's own checks. The core's tests cover the download itself.

## 0.2.2 (2026-11-14)

- The node downloads missing checkpoints at its start, before the models load. It prints Boxer's terms, the files
  and their size first. With all checkpoints there it uses no network and writes no file.
- `--download-only` downloads and checks the checkpoints, then exits 0. It needs no camera, Zenoh router or GPU, and
  `--camera` and `--labels` may be left out. `tools/setup.sh` calls it after the submodule and `uv sync`, and no
  longer needs curl.
- `--no-download` ends a start at once when a checkpoint is missing and names it. It is a read-only parameter.
- `weights.py` is the one list of the checkpoints (names, SHA-256 and the terms moved from `tools/setup.sh`, the
  terms unchanged). `tools/setup.sh` now prints them only when a checkpoint is downloaded, after it has fetched
  Boxer's source.
- Without Boxer's source (the submodule) the node downloads nothing and names `tools/setup.sh`.
- A download goes to `<name>.part` under the lock `ckpts/.download.lock` and gets its final name after the SHA-256
  check.
- The download takes the checkpoints from a fixed commit of `facebook/boxer` (bfba912, where `main` was on
  2026-11-14), no longer from `main`. Hugging Face states the same SHA-256 values for the files there. A download is
  also checked against the exact size.

## 0.2.1 (2026-11-14)

`tools/setup.sh` prints Boxer's terms (CC BY-NC 4.0) before it downloads Boxer's code and checkpoints.

## 0.2.0 (2026-11-13)

Every name in code, on the wire and on disk is `xblick` now. Only the prose name of the project stays Durchblick.
The node needs the core `xblick` 0.5 and does not work with programs of Durchblick 0.4 (`blick`) or older: run every
program of a setup at 0.5.

| What | 0.1.0 | 0.2.0 |
| --- | --- | --- |
| Repository | github.com/xblick/durchblick-boxer | github.com/xblick/xblick-boxer |
| Distribution | `durchblick-boxer` | `xblick-boxer` |
| Python package | `durchblick_boxer` | `xblick_boxer` |
| Commands | `durchblick-boxer` | `xblick-boxer` |
| Program name in log lines, the logger and reports | `durchblick-boxer` | `xblick-boxer` |
| Core dependency | `blick>=0.4,<0.5`, git main of github.com/xblick/durchblick | `xblick>=0.5.0,<0.6`, github.com/xblick/xblick at 45908e6 |
| Core imports and command | `import blick`, `blick rerun`, `blick play`, `blick param` | `import xblick`, `xblick rerun`, `xblick play`, `xblick param` |
| Links to the core's docs | github.com/xblick/durchblick/blob/main/docs/... | github.com/xblick/xblick/blob/main/docs/... |
| Other nodes named in the docs | `durchblick-fast-foundationstereo`, `durchblick-mapping`, `durchblick-nvblox` | `xblick-fast-foundationstereo`, `xblick-mapping`, `xblick-nvblox` |

## 0.1.0

The node as `durchblick-boxer` (Python package `durchblick_boxer`) on the core `blick` 0.4.
