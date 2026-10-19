# PI0.5 SO101 HIL safety invariants

These values are part of the validated real-robot inference behavior. Do not
replace them with distance-dependent 1/2 degree stepping without a new physical
regression evaluation.

- Policy joint step limit: constant **5 degrees per executed action**.
- Calibration-derived soft-limit proposal tolerance: **2 degrees**.
- Every accepted command is still clipped to the true calibration soft limit;
  the 2-degree tolerance does not extend the physical commanded range.
- Normalized PI0.5 actions are clipped to `[-1, 1]` before quantile
  denormalization.
- Unsafe model chunks are discarded and resampled from a fresh observation.
- Policy tracking error up to 2.5 degrees is treated as normal SO101
  gravity/backlash compliance. More than 2.5 degrees with no 0.05-degree
  progress for 3 seconds rejects the current policy chunk instead of terminating
  the collection process.
- Three consecutive unsafe chunks transfer control to human intervention.
- Human teleoperation remains protected by calibration limits and the same
  follower 5-degree relative-target limiter.

The 5-degree/2-degree pair reproduced the successful 30-second PI0.5 rollout.
Reducing the step dynamically near a limit changed the closed-loop timing and
degraded task completion. Values above 5/2 require a deliberate safety review;
they are not the default response to slow task completion.

`record_pi05_human_inloop.sh` rejects a non-5-degree step by default. A deliberate
unvalidated experiment must set `PI05_HIL_ALLOW_UNVALIDATED_LIMITS=1` explicitly.

## USB stability

The robot controllers are QinHeng `1a86:55d3` devices behind two `1a40:0101`
USB 2.0 hubs. Install `config/99-lerobot-usb-power.rules` into
`/etc/udev/rules.d/` to disable autosuspend for the hubs, controllers, and robot
cameras. If re-enumeration continues after this rule is active, replace the
unpowered hub/cable or use a powered hub; software cannot repair an electrical
dropout.
