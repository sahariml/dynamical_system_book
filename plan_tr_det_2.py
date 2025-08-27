import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# -----------------------------
# Grille (tr, det)
# -----------------------------
tr_min, tr_max, n_tr = -4.0, 4.0, 800
det_min, det_max, n_det = -4.0, 4.0, 800
tr_vals = np.linspace(tr_min, tr_max, n_tr)
det_vals = np.linspace(det_min, det_max, n_det)
Tr, Det = np.meshgrid(tr_vals, det_vals)

# -----------------------------
# Discriminant et valeurs propres
# -----------------------------
Delta = Tr**2 - 4.0*Det
is_real = Delta >= 0.0
is_cplx = ~is_real

sqrtDelta = np.zeros_like(Delta)
sqrtDelta[is_real] = np.sqrt(Delta[is_real])
l1 = np.full_like(Tr, np.nan, dtype=float)
l2 = np.full_like(Tr, np.nan, dtype=float)
l1[is_real] = 0.5*(Tr[is_real] + sqrtDelta[is_real])
l2[is_real] = 0.5*(Tr[is_real] - sqrtDelta[is_real])

r = np.full_like(Tr, np.nan, dtype=float)
pos_det_cplx = is_cplx & (Det > 0.0)
r[pos_det_cplx] = np.sqrt(Det[pos_det_cplx])

# -----------------------------
# Zones
# -----------------------------
zones = np.zeros_like(Tr, dtype=int)

# Cas mixtes
zones[is_real & (l1 > 0) & (l1 < 1) & (l2 > 1)] = 1
zones[is_real & (l1 < -1) & (l2 > -1) & (l2 < 0)] = 2
zones[is_real & (l2 < -1) & (l1 > -1) & (l1 < 0)] = 3
zones[is_real & (l1 > 1) & (l2 > 0) & (l2 < 1)] = 4
zones[is_real & (l1 > 1) & (l2 < -1)] = 5
zones[is_real & (l1 > 0) & (l1 < 1) & (l2 < -1)] = 6
zones[is_real & (l1 > 1) & (l2 > -1) & (l2 < 0)] = 10
zones[is_real & (l1 > 0) & (l1 < 1) & (l2 > -1) & (l2 < 0)] = 11
zones[is_real & (l2 > 0) & (l2 < 1) & (l1 > -1) & (l1 < 0)] = 12

# Cas complexes
atol = 1e-3
zones[pos_det_cplx & (np.abs(r - 1.0) <= atol)] = 7
zones[pos_det_cplx & (r < 1.0 - atol)] = 8
zones[pos_det_cplx & (r > 1.0 + atol)] = 9

# Cas "mêmes intervalles"
zones[is_real & (l2 > 1)] = 13
zones[is_real & (l1 < -1)] = 14
zones[is_real & (l2 > 0) & (l1 < 1)] = 15
zones[is_real & (l1 < 0) & (l2 > -1)] = 16

# -----------------------------
# Palette
# -----------------------------
colors = [
    "#f0f0f0", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#ff1493",
    "#00ced1", "#f4a261", "#264653", "#2a9d8f", "#e76f51"
]
cmap = ListedColormap(colors)

# -----------------------------
# Tracé principal
# -----------------------------
fig, ax = plt.subplots(figsize=(9, 9))
im = ax.pcolormesh(Tr, Det, zones, cmap=cmap, shading="auto", rasterized=True)

ax.set_xlabel("tr(A)")
ax.set_ylabel("$\det(A)$")
#ax.set_title("Plan (tr(A), det(A)) – zones des valeurs propres (placement manuel)")
ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlim(tr_min, tr_max)
ax.set_ylim(det_min, det_max)

# -----------------------------
# Lignes de séparation
# -----------------------------
tr_fine = np.linspace(tr_min, tr_max, 3000)
ax.plot(tr_fine, (tr_fine**2)/4, "k--", lw=1.8, label=r"$\Delta=0$")
ax.hlines(1.0, -2, 2, colors="crimson", lw=1.8, linestyles="-.", label=r"$\lambda_{1,2}=e^{\pm i \theta}$")
ax.plot(tr_fine, tr_fine - 1, "b-", lw=1.5, label=r"$\mathrm{tr}(A) - \det(A) - 1=0$ ($\lambda_{1,2}=1$)")
ax.plot(tr_fine, -tr_fine - 1, "g-", lw=1.5, label=r"$\mathrm{tr}(A) + \det(A) + 1=0$ ($\lambda_{1,2}=-1$)")
#ax.hlines(-1.0, tr_min, tr_max, colors="orange", lw=1.5, linestyles=":", label=r"$\det=-1$")

# Points spéciaux
#ax.plot([1, -1, 0], [0, 0, -1], "ko", ms=4)

# -----------------------------
# 📍 ANNOTATIONS MANUELLES – toutes les zones
# -----------------------------
annotations = {
    #1:  r"$0<\lambda_1<1,\ \lambda_2>1$",
    #2:  r"$\lambda_1<-1,\ -1<\lambda_2<0$",
    3:  r"$\lambda_2<-1,\ -1<\lambda_1<0$",
    4:  r"$\lambda_1>1,\ 0<\lambda_2<1$",
    5:  r"$\lambda_1>1,\ \lambda_2<-1$",
    6:  r"$0<\lambda_1<1,\ \lambda_2<-1$",
    ##7:  r"$\lambda_{1,2}\ \text{complexes},\ |\lambda|=1$",
    8:  r"$\lambda_{1,2}\ \text{complexes},\ |\lambda|<1$",
    9:  r"$\lambda_{1,2}\ \text{complexes},\ |\lambda|>1$",
    10: r"$\lambda_1>1,\ -1<\lambda_2<0$",
    11: r"$0<\lambda_1<1,\ -1<\lambda_2<0$",
    #12: r"$0<\lambda_2<1,\ -1<\lambda_1<0$",
    13: r"$\lambda_{1,2}>1$",
    14: r"$\lambda_{1,2}<-1$",
    ##15: r"$0<\lambda_{1,2}<1$",
    ##16: r"$-1<\lambda_{1,2}<0$"
}

# Placement manuel (modifiable)
positions = {
    #1: (2.5, 2.5),
    #2: (-3, 1),
    3: (-3.9, 1.0),
    4: (2.5, 1),
    5: (0.5, -3),
    6: (-3.2, -1.6),
    ##7: (0.5, 3),
    8: (0.1, 0.75),
    9: (-2, 2.5),
    10: (2, -1.6),
    11: (-0.5, -0.3),
    #12: (-0.5, 0.5),
    13: (3.5, 3),
    14: (-3.9, 3),
    ##15: (0.5, 0.8),
    ##16: (-0.5, -0.8),
}

for zone_id, text in annotations.items():
    if zone_id in positions:
        ax.annotate(text, xy=positions[zone_id], fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.7))


#fig, ax = plt.subplots()

# Exemple d'un point important
#ax.plot(1, 0, "ro")

# Annotation avec flèche
ax.annotate(
    "$0<\lambda_{1,2}<1$",   # Texte
    xy=(0.89, 0.18),               # Point visé
    xytext=(2.0, -0.5),           # Position du texte
    arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8)  # Fond blanc au texte
)

ax.annotate(
    "$-1<\lambda_{1,2}<0$",   # Texte
    xy=(-0.89, 0.18),               # Point visé
    xytext=(-2.5, -0.5),           # Position du texte
    arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8)  # Fond blanc au texte
)



#plt.show()

# -----------------------------
# Légende
# -----------------------------
ax.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.show()
