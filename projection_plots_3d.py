import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_3d_categories_tenses(
    target_dirs,
    label_lookup,
    X,
    y,
    vis_num=None,
    x_min=0,
    x_max=55,
    y_min=0,
    y_max=55,
    z_min=0,
    z_max=55,
    elev=20,
    azim=55,
    scale=2,
    model_name=None,
    layer_idx=None,
    plot_dir=None,
):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    color1 = "tab:blue"
    color2 = "darkorange"
    color3 = "firebrick"

    dir1 = target_dirs["past"]["lda"]
    dir2 = target_dirs["present"]["lda"]
    dir3 = target_dirs["future"]["lda"]
    higher_dir = target_dirs["general"]["lda"]

    xaxis = dir1 / dir1.norm()  # normalize PAST dir (xaxis)
    yaxis = (
        dir2 - (dir2 @ xaxis) * xaxis
    )  # make PRESENT dir (yaxis) orthogonal to PAST dir (xaxis)
    yaxis = yaxis / yaxis.norm()  # normalize PRESENT dir (yaxis)
    zaxis = (
        dir3 - (dir3 @ xaxis) * xaxis - (dir3 @ yaxis) * yaxis
    )  # make FUTURE dir (zaxis) orthogonal to PAST dir (xaxis) and PRES dir (yaxis)
    zaxis = zaxis / zaxis.norm()  # normalize FUTURE dir (zaxis)

    axes = torch.stack([xaxis, yaxis, zaxis], dim=1)

    g1 = torch.tensor(X[np.isin(y, label_lookup["tense"]["past"])])
    g2 = torch.tensor(X[np.isin(y, label_lookup["tense"]["present"])])
    g3 = torch.tensor(X[np.isin(y, label_lookup["tense"]["future"])])
    g = torch.tensor(X)

    proj1 = (g1 @ axes).cpu().numpy()
    proj2 = (g2 @ axes).cpu().numpy()
    proj3 = (g3 @ axes).cpu().numpy()
    proj = (g @ axes).cpu().numpy()

    P1 = (dir1 @ axes).cpu().numpy()
    P2 = (dir2 @ axes).cpu().numpy()
    P3 = (dir3 @ axes).cpu().numpy()
    P4 = (higher_dir @ axes).cpu().numpy()

    # midpoints
    ax.scatter(P1[0], P1[1], P1[2], color=color1, s=100)
    ax.scatter(P2[0], P2[1], P2[2], color=color2, s=100)
    ax.scatter(P3[0], P3[1], P3[2], color=color3, s=100)
    ax.scatter(P4[0], P4[1], P4[2], color="black", s=100)

    # triangle outer
    verts = [
        list(zip([P1[0], P2[0], P3[0]], [P1[1], P2[1], P3[1]], [P1[2], P2[2], P3[2]]))
    ]
    triangle = Poly3DCollection(
        verts, alpha=0.2, linewidths=1, linestyle="--", edgecolors="k"
    )
    triangle.set_facecolor("yellow")
    ax.add_collection3d(triangle)

    # lines to origin
    ax.quiver(0, 0, 0, P1[0], P1[1], P1[2], color=color1, arrow_length_ratio=0.01)
    ax.quiver(0, 0, 0, P2[0], P2[1], P2[2], color=color2, arrow_length_ratio=0.01)
    ax.quiver(0, 0, 0, P3[0], P3[1], P3[2], color=color3, arrow_length_ratio=0.01)

    # all data points
    scatter1 = ax.scatter(
        proj1[:, 0], proj1[:, 1], proj1[:, 2], c=color1, s=3.05, label="past"
    )
    scatter2 = ax.scatter(
        proj2[:, 0], proj2[:, 1], proj2[:, 2], c=color2, s=3.05, label="present"
    )
    scatter3 = ax.scatter(
        proj3[:, 0], proj3[:, 1], proj3[:, 2], c=color3, s=3.05, label="future"
    )
    scatter = ax.scatter(
        proj[:, 0], proj[:, 1], proj[:, 2], c="gray", s=3.05, alpha=0.03
    )

    # text boxes
    # scale = 2.038 #1.2
    ax.text(
        P1[0] * scale + 2,
        P1[1] * scale,
        P1[2] * scale,
        "past",
        bbox=dict(facecolor=color1, edgecolor="cornflowerblue", alpha=0.5),
    )
    ax.text(
        P2[0] * scale + 0.5,
        P2[1] * scale + 0.5,
        P2[2] * scale,
        "present",
        bbox=dict(facecolor=color2, edgecolor="sandybrown", alpha=0.5),
    )
    ax.text(
        P3[0] * scale,
        P3[1] * scale,
        P3[2] * scale,
        "future",
        bbox=dict(facecolor=color3, edgecolor="darkred", alpha=0.5),
    )

    normal_vector = np.cross(P2 - P1, P3 - P1)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    normal_mag = P1 @ normal_vector
    normal_vector = normal_vector * normal_mag

    P1_normal = P1 - normal_vector
    P2_normal = P2 - normal_vector
    P3_normal = P3 - normal_vector

    # triangle inner
    ax.quiver(
        normal_vector[0],
        normal_vector[1],
        normal_vector[2],
        P1_normal[0],
        P1_normal[1],
        P1_normal[2],
        color=color1,
        linestyle="--",
        arrow_length_ratio=0.01,
    )
    ax.quiver(
        normal_vector[0],
        normal_vector[1],
        normal_vector[2],
        P2_normal[0],
        P2_normal[1],
        P2_normal[2],
        color=color2,
        linestyle="--",
        arrow_length_ratio=0.01,
    )
    ax.quiver(
        normal_vector[0],
        normal_vector[1],
        normal_vector[2],
        P3_normal[0],
        P3_normal[1],
        P3_normal[2],
        color=color3,
        linestyle="--",
        arrow_length_ratio=0.01,
    )
    ax.quiver(
        0,
        0,
        0,
        normal_vector[0],
        normal_vector[1],
        normal_vector[2],
        color="gray",
        linestyle="--",
        arrow_length_ratio=0.01,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.view_init(elev=elev, azim=azim)

    if plot_dir:
        fig.savefig(
            f"{plot_dir}/{model_name}_L{layer_idx}_3d_tenses_v{vis_num}.pdf",
            format="pdf",
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


def plot_3d_categories_aspect(
    target_dirs,
    label_lookup,
    X,
    y,
    vis_num=None,
    x_min=-21.3,
    x_max=6.7,
    y_min=-127.7,
    y_max=4.5,
    z_min=-18.7,
    z_max=14.5,
    elev=19.5,
    azim=-141.6,
    scale=2,
    scale2=2,
    model_name=None,
    layer_idx=None,
    plot_dir=None,
):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    color1 = "tab:blue"
    color2 = "mediumaquamarine"
    color3 = "firebrick"
    color4 = "darkorange"

    # Categories
    cat1 = "simple"
    cat2 = "progressive"
    cat3 = "perfect"
    cat4 = "perfect_progressive"

    # Get the directions from target_dirs
    dir1 = target_dirs[cat1]["lda"]
    dir2 = target_dirs[cat2]["lda"]
    dir3 = target_dirs[cat3]["lda"]
    dir4 = target_dirs[cat4]["lda"]

    # Calculate the axes
    xaxis = (dir2 - dir1) / (dir2 - dir1).norm()
    yaxis = dir3 - dir1 - (dir3 - dir1) @ xaxis * xaxis
    yaxis = yaxis / yaxis.norm()
    zaxis = (
        (dir4 - dir1) - (dir4 - dir1) @ xaxis * xaxis - (dir4 - dir1) @ yaxis * yaxis
    )
    zaxis = zaxis / zaxis.norm()
    axes = torch.stack([xaxis, yaxis, zaxis], dim=1)

    # Get the data for each category
    g1 = torch.tensor(X[np.isin(y, label_lookup["aspect"][cat1])])
    g2 = torch.tensor(X[np.isin(y, label_lookup["aspect"][cat2])])
    g3 = torch.tensor(X[np.isin(y, label_lookup["aspect"][cat3])])
    g4 = torch.tensor(X[np.isin(y, label_lookup["aspect"][cat4])])

    proj1 = (g1 @ axes).cpu().numpy()
    proj2 = (g2 @ axes).cpu().numpy()
    proj3 = (g3 @ axes).cpu().numpy()
    proj4 = (g4 @ axes).cpu().numpy()

    P1 = (dir1 @ axes).cpu().numpy()
    P2 = (dir2 @ axes).cpu().numpy()
    P3 = (dir3 @ axes).cpu().numpy()
    P4 = (dir4 @ axes).cpu().numpy()

    # all data points
    scatter1 = ax.scatter(
        proj1[:, 0], proj1[:, 1], proj1[:, 2], c=color1, s=3.05, label=cat1
    )
    scatter2 = ax.scatter(
        proj2[:, 0], proj2[:, 1], proj2[:, 2], c=color2, s=3.05, label=cat2
    )
    scatter3 = ax.scatter(
        proj3[:, 0], proj3[:, 1], proj3[:, 2], c=color3, s=3.05, label=cat3
    )
    scatter = ax.scatter(
        proj4[:, 0], proj4[:, 1], proj4[:, 2], c=color4, s=3.05, label=cat4
    )

    # midpoints
    ax.scatter(P1[0], P1[1], P1[2], color=color1, s=100)
    ax.scatter(P2[0], P2[1], P2[2], color=color2, s=100)
    ax.scatter(P3[0], P3[1], P3[2], color=color3, s=100)
    ax.scatter(P4[0], P4[1], P4[2], color=color4, s=100)

    # Draw triangles
    verts1 = [
        list(zip([P1[0], P2[0], P3[0]], [P1[1], P2[1], P3[1]], [P1[2], P2[2], P3[2]]))
    ]
    triangle1 = Poly3DCollection(
        verts1, alpha=0.1, linewidths=1, linestyle="--", edgecolors="k"
    )
    triangle1.set_facecolor("yellow")
    ax.add_collection3d(triangle1)

    verts2 = [
        list(zip([P1[0], P2[0], P4[0]], [P1[1], P2[1], P4[1]], [P1[2], P2[2], P4[2]]))
    ]
    triangle2 = Poly3DCollection(
        verts2, alpha=0.1, linewidths=1, linestyle="--", edgecolors="k"
    )
    triangle2.set_facecolor("yellow")
    ax.add_collection3d(triangle2)

    verts3 = [
        list(zip([P1[0], P3[0], P4[0]], [P1[1], P3[1], P4[1]], [P1[2], P3[2], P4[2]]))
    ]
    triangle3 = Poly3DCollection(
        verts3, alpha=0.1, linewidths=1, linestyle="--", edgecolors="k"
    )
    triangle3.set_facecolor("yellow")
    ax.add_collection3d(triangle3)

    verts4 = [
        list(zip([P2[0], P3[0], P4[0]], [P2[1], P3[1], P4[1]], [P2[2], P3[2], P4[2]]))
    ]
    triangle4 = Poly3DCollection(
        verts4, alpha=0.1, linewidths=1, linestyle="--", edgecolors="k"
    )
    triangle4.set_facecolor("yellow")
    ax.add_collection3d(triangle4)

    ax.quiver(0, 0, 0, P1[0], P1[1], P1[2], color=color1, arrow_length_ratio=0.01)
    ax.quiver(0, 0, 0, P2[0], P2[1], P2[2], color=color2, arrow_length_ratio=0.01)
    ax.quiver(0, 0, 0, P3[0], P3[1], P3[2], color=color3, arrow_length_ratio=0.01)
    ax.quiver(0, 0, 0, P4[0], P4[1], P4[2], color=color4, arrow_length_ratio=0.01)

    # text boxes
    # scale = 2.038 #1.2
    ax.text(
        P1[0] * scale + 2,
        P1[1] * scale,
        P1[2] * scale,
        cat1,
        bbox=dict(facecolor=color1, edgecolor="cornflowerblue", alpha=0.5),
    )
    ax.text(
        P2[0] * scale + 1,
        P2[1] * scale + 4,
        P2[2] * scale,
        cat2,
        bbox=dict(facecolor=color2, edgecolor="tab:cyan", alpha=0.5),
    )
    ax.text(
        P3[0] * scale,
        P3[1] * scale,
        P3[2] * scale,
        cat3,
        bbox=dict(facecolor=color3, edgecolor="darkred", alpha=0.5),
    )
    ax.text(
        P4[0] * scale2 + 2,
        P4[1] * scale2,
        P4[2] * scale2 - 1,
        cat4,
        bbox=dict(facecolor=color4, edgecolor="sandybrown", alpha=0.2),
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.view_init(elev=elev, azim=azim)

    if plot_dir:
        fig.savefig(
            f"{plot_dir}/{model_name}_L{layer_idx}_3d_aspect_v{vis_num}.pdf",
            format="pdf",
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
