# Upstream baselines

The patches were exported from working trees against these exact commits on 2026-10-19. Upstream projects retain their own licenses and attribution.

| Component | Upstream | Commit | Local changes |
| --- | --- | --- | --- |
| LeRobot | https://github.com/JoyandAI/lerobot | `072e8355fbb9a7b520915d44877aa876794efb31` | `patches/lerobot-smolvla-awr.patch` |
| Evo-RL | https://github.com/MINT-SJTU/Evo-RL | `6f2db449a21e1bac750b996f2e27cac6739aa63f` | `patches/evo-rl-pi05-hil.patch` and `overlays/evo_rl/` |
| RoboTwin | https://github.com/RoboTwin-Platform/RoboTwin | `96c1feab536306b50c26af200044fcdf126e8904` | `patches/robotwin-memory-data.patch` |
| RoboTwin XPolicyLab submodule | Pinned by RoboTwin | `c37109c500be67d0dea6b36bf7337bbd26e763cd` | Not included in the RoboTwin patch |

Apply each patch only to its matching commit, in a separate clone. For example:

```bash
git clone https://github.com/MINT-SJTU/Evo-RL.git
cd Evo-RL
git checkout 6f2db449a21e1bac750b996f2e27cac6739aa63f
git apply --check ../robot-memory-lab/patches/evo-rl-pi05-hil.patch
git apply ../robot-memory-lab/patches/evo-rl-pi05-hil.patch
cp -a ../robot-memory-lab/overlays/evo_rl/. .
```

Use the same `git apply --check` / `git apply` sequence for the LeRobot and RoboTwin patches. Do not apply a patch to a newer or already-modified checkout without reviewing the resulting diff.

`overlays/evo_rl/` contains only the two newly added HIL files. The untracked `modeling_pi05.py.pre_padding_mask` in the source checkout was a backup and is intentionally excluded.

