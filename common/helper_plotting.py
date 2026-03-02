import matplotlib.pyplot as plt
import numpy as np


def plot_1d_seq(trj, pip, title, **kwargs):
    trj_np = trj.detach().cpu().numpy() if hasattr(trj, "detach") else np.array(trj)

    fig, ax = plt.subplots(figsize=(6, 4))
    # Plot all dimensions in one go using transpose
    T = trj_np.shape[0]
    x = np.arange(T)
    idx = x.shape[0]
    if "idx" in kwargs:
        idx = kwargs["idx"]
    x = x[:idx]
    # Stack predictions and targets to distinguish lines
    ax.plot(x[:, None], trj_np[:idx], "-", color="red", label="Missile")
    if "position" in title:
        ax.plot(x[-1], pip, "x", color="green", label="PIP")

    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img_np = buf[:, :, :3]
    plt.close()
    return img_np


def plot_3d_trj_gen(trj_out, pip, title, mod=" "):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        trj_out[:, 0],
        trj_out[:, 1],
        trj_out[:, 2],
        "-",
        color="red",
        label="Missile",
        linewidth=2,
    )
    if mod == "position":
        ax.plot(
            pip[0],
            pip[1],
            pip[2],
            "X",
            color="green",
            label="PIP",
            linewidth=6,
        )

    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.set_zlabel("Z", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend()

    ax.view_init(elev=20, azim=120)
    plt.tight_layout()

    # Render the canvas
    fig.canvas.draw()

    # Get RGBA buffer from canvas
    buf = np.asarray(fig.canvas.buffer_rgba())  # <-- modern way

    # buf is (H, W, 4) because of alpha channel (RGBA), we drop the alpha
    img_np = buf[:, :, :3]  # Keep only RGB

    plt.close(fig)
    return img_np


def plot_3d_trajectory(trj_out, trj_true, title):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        trj_out[:, 0],
        trj_out[:, 1],
        trj_out[:, 2],
        "-",
        color="red",
        label="Missile",
        linewidth=2,
    )
    ax.plot(
        trj_true[:, 0],
        trj_true[:, 1],
        trj_true[:, 2],
        "-",
        color="blue",
        label="Target",
        linewidth=2,
    )
    ax.plot(
        trj_out[-1, 0],
        trj_out[-1, 1],
        trj_out[-1, 2],
        "x",
        color="green",
        label="PIP",
        linewidth=6,
    )

    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.set_zlabel("Z", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend()

    ax.view_init(elev=20, azim=120)
    plt.tight_layout()

    # Render the canvas
    fig.canvas.draw()

    # Get RGBA buffer from canvas
    buf = np.asarray(fig.canvas.buffer_rgba())  # <-- modern way

    # buf is (H, W, 4) because of alpha channel (RGBA), we drop the alpha
    img_np = buf[:, :, :3]  # Keep only RGB

    plt.close(fig)
    return img_np


def plot_1d_from_pos_vel(trajectory, pip, writer, name="pos", tag_base="", **kwargs):
    img = plot_1d_seq(trajectory[:, 0], pip[0], f"{name}_x_missile_to_pip", **kwargs)
    writer.add_image(
        tag=f"{tag_base}/{name}_x_missile_to_pip",
        global_step=0,
        img_tensor=img,
        dataformats="HWC",
    )
    img = plot_1d_seq(trajectory[:, 1], pip[1], f"{name}_y_missile_to_pip", **kwargs)
    writer.add_image(
        tag=f"{tag_base}/{name}_y_missile_to_pip",
        global_step=0,
        img_tensor=img,
        dataformats="HWC",
    )
    img = plot_1d_seq(trajectory[:, 2], pip[2], f"{name}_z_missile_to_pip", **kwargs)
    writer.add_image(
        tag=f"{tag_base}/{name}_z_missile_to_pip",
        global_step=0,
        img_tensor=img,
        dataformats="HWC",
    )
