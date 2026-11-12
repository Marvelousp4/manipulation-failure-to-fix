# Protocol

- Freeze the checkpoint; evaluate 50 identical seeds in three separate conditions.
- Clean; additive independent Gaussian state noise (sigma 0.01); FIFO action delay (2 control steps).
- Noise affects every floating state coordinate in native units, without normalization/clipping.
- Delay starts with zero commands and resets each episode. Controller: `pd_ee_delta_pos`.
- CPU PhysX, 100-step horizon. Report success at any step and at the last step; termination is not success.
- Verify exact initial observation hashes across conditions. These do not certify every hidden simulator state.
- Report Wilson 95% intervals and paired regressions/improvements; unknown success stays null.
- Fixed final training checkpoint; evaluation seeds 10000–10049 differ from demonstration seeds 0–999.
- One training seed/task/budget supports a preliminary baseline, not broad robustness claims.
- Environment factory and policy loader isolate simulator-specific code; future backends use the same evaluator.
