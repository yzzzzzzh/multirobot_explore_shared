# Validation record

## Clean-clone acceptance run

The focused repository was cloned into a separate directory at commit
`8d3a2501893a61ebb2da5314ce914d86eba4987f` and run with:

```bash
bash scripts/verify_repository.sh
bash scripts/preflight.sh
bash scripts/run_legged_dual_exploration.sh 200 cleanclone_acceptance_200s_v3
```

The run requested 200 simulation seconds. The first-to-last retained samples
span 198.3995 seconds; this is expected because recording starts and stops
between sampled messages. The recorder itself reached the requested simulated
duration before finishing.

| Check | Result |
|---|---:|
| bot1 ground-truth distance | 164.666 m |
| bot2 ground-truth distance | 182.413 m |
| final planar coverage | 44.985% |
| observed navigable area | 1982.25 m2 |
| minimum inter-robot distance | 6.299 m |
| inter-robot contact | false |
| emergency stop | false |
| monitor warnings | 0 |
| bot1 SLAM valid fraction | 1.0 |
| bot2 SLAM valid fraction | 1.0 |
| bot1 trajectory-aligned ATE RMSE | 0.218 m |
| bot2 trajectory-aligned ATE RMSE | 0.337 m |

Both generated MP4 files were fully decoded with FFmpeg without errors, and
all entries in the run's `SHA256SUMS` passed. The run also produced valid NPZ,
summary, analysis, planar-coverage and HGrid diagnostic artifacts. Containers
were stopped after completion.

This is operational validation of the clean-clone execution path. It is not a
claim that a fresh source build is bit-identical to the historical containers,
nor that a 200-s stochastic trajectory must match the archived 700-s run
sample-for-sample.
