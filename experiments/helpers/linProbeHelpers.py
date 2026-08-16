import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import torch

def wm_transform(preprocessor, key):
    def _apply(img):
        sample = {key: img}
        preprocessor(sample)
        return sample[key]
    return _apply

def make_pusht_state(pusher_xy, block_xy, block_theta=0.0, velocity=(0.0, 0.0)):
    """Build a 7-d Push-T state vector."""
    px, py = pusher_xy
    bx, by = block_xy
    vx, vy = velocity
    return np.array([px, py, bx, by, block_theta, vx, vy], dtype=np.float64)


def visualize_pusht_layout(
    start_state,
    goal_state,
    *,
    figsize=(10, 4),
    show_coords=True,
    hazard_box=None,  # optional: (x_min, x_max, y_min, y_max)
):
    """
    Reset Push-T to start_state / goal_state and show start vs goal images.

    hazard_box: optional rectangle in block-coordinate space for planning sketches
    """
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    obs, info = env.reset(options={"state": start_state, "goal_state": goal_state})

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(env.render())
    axes[0].set_title("Start")

    axes[1].imshow(info["goal"])
    axes[1].set_title("Goal")

    if show_coords:
        sb = obs["state"][2:4]
        gb = goal_state[2:4]
        sp = obs["state"][:2]
        gp = goal_state[:2]
        axes[0].set_xlabel(f"pusher {sp.round(0)}, block {sb.round(0)}")
        axes[1].set_xlabel(f"pusher {gp.round(0)}, block {gb.round(0)}")

    if hazard_box is not None:
        from matplotlib.patches import Rectangle
        x0, x1, y0, y1 = hazard_box
        # Arena coords match image coords: (0, 0) = top-left
        scale = 224 / 512
        for ax in axes:
            ax.add_patch(Rectangle(
                (x0 * scale, y0 * scale),
                (x1 - x0) * scale,
                (y1 - y0) * scale,
                fill=False, edgecolor="red", linewidth=2, linestyle="--",
            ))

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    summary = {
        "start_pusher_xy": obs["state"][:2].copy(),
        "start_block_xy": obs["state"][2:4].copy(),
        "goal_pusher_xy": goal_state[:2].copy(),
        "goal_block_xy": goal_state[2:4].copy(),
        "start_block_theta": obs["state"][4],
        "goal_block_theta": goal_state[4],
    }
    env.close()
    return summary

def make_on_step(frames, states):
    def on_step(w):
        frame = np.asarray(w.infos["pixels"][0, -1]).copy()
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        frames.append(frame)
        states.append(w.infos["state"][0, 0].copy())
    return on_step

def animate_frames(frames, interval_ms=100, figsize=(4, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    im = ax.imshow(frames[0])
    title = ax.set_title("step 0")
    def update(i):
        im.set_array(frames[i])
        title.set_text(f"step {i}")
        return [im, title]
    anim = FuncAnimation(
        fig, update, frames=len(frames), interval=interval_ms, blit=True
    )
    plt.close(fig)
    return HTML(anim.to_jshtml())

def plot_trajectory(states, start_state, goal_state, hazard_box=None):
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(states[:, 2], states[:, 3], "o-", ms=3, label="block path")
    ax.plot(states[:, 0], states[:, 1], ".-", ms=4, alpha=0.6, label="pusher path")

    ax.scatter(start_state[2], start_state[3], c="tab:green", s=80, label="start block")
    ax.scatter(goal_state[2], goal_state[3], c="tab:red", s=80, label="goal block")
    ax.scatter(states[0, 0], states[0, 1], c="tab:blue", s=40, label="start pusher")

    if hazard_box is not None:
        x0, x1, y0, y1 = hazard_box
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=True, alpha=0.2, color="red", label="hazard",
        ))

    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # match image coords if needed; remove if plot looks flipped
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    ax.set_title("Rollout trajectory")
    plt.tight_layout()
    plt.show()

def box_penetration(xy, hazard_box):
    """
    xy: (..., 2) block positions
    returns: (...,) penetration depth (>=0 inside, 0 outside)
    """
    x0, x1, y0, y1 = hazard_box
    dx = torch.minimum(xy[..., 0] - x0, x1 - xy[..., 0])
    dy = torch.minimum(xy[..., 1] - y0, y1 - xy[..., 1])
    return torch.clamp(torch.minimum(dx, dy), min=0.0)


def hazard_cost_from_emb(predicted_emb, hist_len, probe_w, probe_b, hazard_box):
    """
    predicted_emb: (B, S, T, D)
    Sum penetration over *future* predicted steps (after context frames).
    """
    future = predicted_emb[:, :, hist_len:, :]  # (B, S, Tp, D)
    B, S, Tp, D = future.shape
    flat = future.reshape(B * S * Tp, D)
    xy = flat @ probe_w + probe_b  # (B*S*Tp, 2)
    pen = box_penetration(xy, hazard_box).reshape(B, S, Tp)
    return pen.sum(dim=-1)  # (B, S)


class HazardAugmentedCostModel(torch.nn.Module):
    """C = goal_MSE + λ * sum_t penetration(probe(z_t))."""

    def __init__(self, base_model, ridge_probe, hazard_box, lam: float):
        super().__init__()
        self.base = base_model
        self.hazard_box = tuple(map(float, hazard_box))
        self.lam = float(lam)
        # Ridge: y = z @ coef_.T + intercept_
        self.register_buffer(
            "probe_w",
            torch.as_tensor(ridge_probe.coef_.T, dtype=torch.float32),
        )
        self.register_buffer(
            "probe_b",
            torch.as_tensor(ridge_probe.intercept_, dtype=torch.float32),
        )
        self.cost_history = []

    def get_cost(self, info_dict, action_candidates):
        goal_cost = self.base.get_cost(info_dict, action_candidates)

        hist_len = info_dict["pixels"].shape[2]
        hazard_cost = hazard_cost_from_emb(
            info_dict["predicted_emb"],
            hist_len,
            self.probe_w,
            self.probe_b,
            self.hazard_box,
        )

        self.last_goal_cost = goal_cost.detach().cpu()
        self.last_hazard_cost = hazard_cost.detach().cpu()
        self.cost_history.append({
            "goal_min": goal_cost.min().item(),
            "goal_median": goal_cost.median().item(),
            "goal_max": goal_cost.max().item(),
            "hazard_min": hazard_cost.min().item(),
            "hazard_median": hazard_cost.median().item(),
            "hazard_max": hazard_cost.max().item(),
            "hazard_nonzero": (
                (hazard_cost > 0).float().mean().item()
            ),
        })
        return goal_cost + self.lam * hazard_cost


def real_violation_stats(states, hazard_box, entity="pusher"):
    x0, x1, y0, y1 = hazard_box

    if entity == "pusher":
        xy = states[:, 0:2]
    elif entity == "block":
        xy = states[:, 2:4]
    else:
        raise ValueError("entity must be 'pusher' or 'block'")

    inside = (
        (xy[:, 0] >= x0) & (xy[:, 0] <= x1) &
        (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    )
    return {
        "any_violation": bool(inside.any()),
        "n_violating_steps": int(inside.sum()),
        "frac_violating": float(inside.mean()),
    }

def summarize_cem_history(cost_model, n_steps=30, verbose=True):
    history = cost_model.cost_history
    summaries = []

    for solve_idx, start in enumerate(range(0, len(history), n_steps)):
        solve = history[start:start + n_steps]
        if not solve:
            continue

        first = solve[0]
        last = solve[-1]

        summary = {
            "solve": solve_idx + 1,
            "first_hazard_max": first["hazard_max"],
            "first_hazard_nonzero": first["hazard_nonzero"],
            "final_hazard_max": last["hazard_max"],
            "final_hazard_nonzero": last["hazard_nonzero"],
            "first_goal_median": first["goal_median"],
            "final_goal_median": last["goal_median"],
        }
        summaries.append(summary)

        if verbose:
            print(f"\nCEM solve {solve_idx + 1}")
            print(
                f"  first: goal median={first['goal_median']:.3f}, "
                f"hazard max={first['hazard_max']:.3f}, "
                f"hazard nonzero={first['hazard_nonzero']:.3f}"
            )
            print(
                f"  final: goal median={last['goal_median']:.3f}, "
                f"hazard max={last['hazard_max']:.3f}, "
                f"hazard nonzero={last['hazard_nonzero']:.3f}"
            )

    return summaries