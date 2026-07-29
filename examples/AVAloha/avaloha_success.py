"""Pose-based, orientation-aware success predicates for AV-ALOHA eval.

Why not the env's ``info["is_success"]`` (``reward == max_reward``)?  It only
checks that the hidden goal *marker* geoms touch, and that check is **blind to
orientation** and uses wide markers on free-floating bodies. Probing the sim
shows ``is_success`` fires for clearly-wrong poses:

* ``insert_peg``  -- peg rotated 90 deg (lying across the hole) or tilted, as
  long as its center sits at the hole center -> reward 4.
* ``slot_insertion`` -- stick laid across the slot, tilted, or only partly
  lowered (+3 cm) -> reward 4.
* ``tube_transfer`` -- ball resting on the floor directly under the tube
  (z 0.005 m) -> reward 3.

Scalar marker-center *distance* (an earlier attempt) shares the same blind
spots, since a mis-oriented object can still be centered. So instead we express
the moving object in the **socket's local frame** and require it to be genuinely
*contained*: aligned axis + small lateral offset + correct insertion depth (and,
for the tube, the receiver standing upright with the ball elevated inside the
bore). The deploy loop additionally requires the predicate to hold for N
consecutive control steps. All thresholds were calibrated by probing seated vs
adversarial poses directly in MuJoCo (scratch probes).

``hook_package`` / ``sew_needle`` stay on the raw contact criterion for now
(their two markers are both small, so ``is_success`` already needs them nearly
coincident); upgrade them here if they show false positives.
"""

import numpy as np


# Thresholds calibrated on real pins-off rollouts (markers ghosted, so objects
# fully seat as in the training data) and cross-checked against the rendered
# video captions (each frame prints its live metrics).

# ---- insert_peg (socket = hole, moving = peg; channel axis = hole local x) ----
PEG_ALIGN_MIN = 0.94  # |peg_x . hole_x| (cos ~20 deg): reject perpendicular/tilted
PEG_DEPTH_MAX = 0.035  # |along-axis offset| (m); 0 == fully seated (traces: 0.00-0.03)
PEG_LAT_MAX = 0.015  # lateral offset from the channel axis (cavity inner ~0.018)

# ---- slot_insertion (socket = slot, moving = stick; stick lies along slot x) --
SLOT_XALIGN_MIN = 0.94  # stick long axis aligned with the slot channel
SLOT_LEVEL_MIN = 0.94  # stick stays level (its z aligned with slot z)
SLOT_LAT_Y_MAX = 0.012  # centered between the slot bars (seated traces: |y|<0.005)
SLOT_Z_SEAT_MAX = 0.012  # fully lowered into the groove (reject partial/lifted)
SLOT_X_MAX = 0.08  # allowed shift along the channel (still inside the slot)

# ---- tube_transfer (socket = tube2, moving = ball; bore along tube2 local z) --
# "In the tube" = ball inside the bore: laterally centered AND within the tube
# height. Calibrated against the rendered videos: a ball resting at the bottom of
# the (usually lifted & tilted) tube reads lat<=0.028 and relz ~ -0.02 (slightly
# BELOW the tube body origin) -- NOT positive. A ball that fell OUT drops to
# relz ~ -0.10..-0.19 (and/or large lat), so the relz floor separates them. The
# `--success-hold` (>=15) requirement then rejects the brief pass-through while
# the ball is poured (transient ~3-5 steps) vs a ball that settles in (~40-50).
TUBE_BORE_MAX = 0.03  # lateral offset within the bore (square inner ~0.0115, +tilt)
TUBE_Z_LO = -0.04  # ball resting at the tube bottom is slightly negative; reject fell-out (<-0.04)
TUBE_Z_HI = 0.09  # below the tube rim (~0.10); above this it is sitting on top
TUBE_UPRIGHT_MIN = 0.9  # tube2 standing upright (reject tipped-over receiver)

# ---- hook_package / sew_needle: ghosted markers overlap when hooked/threaded ---
HOOK_MARK_MAX = 0.025  # |pin-package - pin-hook| when the package is on the hook
SEW_MARK_MAX = 0.020  # |pin-needle - pin-wall| when the needle is threaded
# NOTE: hook/sew tols are geometry-derived (this checkpoint never succeeds them,
# so there is no positive data to calibrate against yet).


def _Rmat(env, body):
    return np.asarray(env._physics.named.data.xmat[body]).reshape(3, 3)


def _geom_xpos(env, name):
    return np.asarray(env._physics.named.data.geom_xpos[name])


def _marker_dist(env, a, b):
    return float(np.linalg.norm(_geom_xpos(env, a) - _geom_xpos(env, b)))


def _xpos(env, body):
    return np.asarray(env._physics.named.data.xpos[body])


def _rel(env, moving, socket):
    """Position of `moving` in `socket` frame, and `moving`'s axes in `socket` frame."""
    Rs = _Rmat(env, socket)
    p_rel = Rs.T @ (_xpos(env, moving) - _xpos(env, socket))
    axes = Rs.T @ _Rmat(env, moving)  # columns: moving x,y,z expressed in socket frame
    return p_rel, axes


def _insert_peg(env, info):
    p, ax = _rel(env, "peg", "hole")
    align = abs(ax[0, 0])  # peg x-axis . hole x-axis
    lat = float(np.hypot(p[1], p[2]))
    depth = abs(float(p[0]))
    ok = align > PEG_ALIGN_MIN and depth < PEG_DEPTH_MAX and lat < PEG_LAT_MAX
    return ok, f"al={align:.2f} dx={depth:.3f} lat={lat:.3f}"


def _slot_insertion(env, info):
    p, ax = _rel(env, "stick", "slot")
    xalign = abs(ax[0, 0])  # stick x . slot x
    level = abs(ax[2, 2])  # stick z . slot z
    ok = (
        xalign > SLOT_XALIGN_MIN
        and level > SLOT_LEVEL_MIN
        and abs(float(p[1])) < SLOT_LAT_Y_MAX
        and abs(float(p[2])) < SLOT_Z_SEAT_MAX
        and abs(float(p[0])) < SLOT_X_MAX
    )
    return ok, f"al={xalign:.2f} lv={level:.2f} y={p[1]:+.3f} z={p[2]:+.3f}"


def _tube_transfer(env, info):
    Rt = _Rmat(env, "tube2")
    p = Rt.T @ (_xpos(env, "ball") - _xpos(env, "tube2"))  # ball in tube2 frame
    lat = float(np.hypot(p[0], p[1]))
    up = float(Rt[2, 2])
    ok = lat < TUBE_BORE_MAX and TUBE_Z_LO < float(p[2]) < TUBE_Z_HI and up > TUBE_UPRIGHT_MIN
    return ok, f"lat={lat:.3f} z={p[2]:+.3f} up={up:.2f}"


def _hook_package(env, info):
    d = _marker_dist(env, "pin-package", "pin-hook")
    return d < HOOK_MARK_MAX, f"d={d:.3f}"


def _sew_needle(env, info):
    d = _marker_dist(env, "pin-needle", "pin-wall")
    return d < SEW_MARK_MAX, f"d={d:.3f}"


def _raw(env, info):
    return bool(info.get("is_success")), ""


_SEATED = {
    "insert_peg": _insert_peg,
    "slot_insertion": _slot_insertion,
    "tube_transfer": _tube_transfer,
    "hook_package": _hook_package,
    "sew_needle": _sew_needle,
}


def evaluate(task, env, info):
    """Return ``(seated: bool, detail: str)`` for ``task``.

    ``env`` must be the unwrapped ``GuidedVisionEnv`` (``gym_env.unwrapped``).
    ``detail`` is a short human-readable string for video captions / logs.
    """
    return _SEATED.get(task, _raw)(env, info)


def disable_marker_collisions(env):
    """Make the goal *marker* geoms (group 3, the ``gap=100`` 'pin' sensors)
    non-colliding, restoring their intended ghost behavior.

    These markers were authored as sensor-only ghosts (``gap=100`` => contact
    detected but force suppressed). In MuJoCo 3.10 ``gap > margin`` is ignored,
    so the markers become SOLID and block the task physics — the peg cannot
    reach the hole center, and the ball cannot fall into the tube (it perches on
    top of the pin). That diverges from the training data (collected where the
    markers were ghosts: ball falls to the tube bottom, peg fully inserts).
    Turning the markers non-colliding (``contype = conaffinity = 0``) restores
    training-consistent physics. Success is then detected from object poses
    (``evaluate``), not from marker contact. Returns the disabled geom names.
    """
    m = env._physics.model.ptr
    disabled = []
    for gid in range(m.ngeom):
        # The marker 'pin' sensors are the only geoms authored with gap=100;
        # group=3 is NOT unique (table, grippers, robot geoms use it too).
        if float(m.geom_gap[gid]) > 1.0:
            m.geom_contype[gid] = 0
            m.geom_conaffinity[gid] = 0
            disabled.append(env._physics.model.id2name(gid, "geom"))
    env._physics.forward()
    return disabled
