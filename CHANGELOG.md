# Changelog

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
