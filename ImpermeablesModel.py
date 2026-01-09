import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.ndimage import sum as ndi_sum

plt.rcParams.update({
    'axes.labelsize': 45,
    'axes.titlesize': 50,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'legend.fontsize': 25
})


M = 2000
L_list = [10, 50, 100, 200]
p = 0.58          # try 0.50, 0.54, 0.58 etc.
a = 1.2           

# prob of impermeables - change these to alter q value in graphs. 
q_list = [0.4,0.42, 0.44, 0.45, 0.46]   


structure4 = np.array([[0,1,0],
                       [1,1,1],
                       [0,1,0]])

styles = {
    10:  ("orange", "-"),
    50:  ("red", "-"),
    100: ("blue", "--"),
    200: ("green", "-.")
}

#spanning check
def spans(lw):
    """Return True if there exists a cluster spanning left-right OR top-bottom."""
    # label 0 is background, ignore it
    left   = set(np.unique(lw[:, 0])) - {0}
    right  = set(np.unique(lw[:, -1])) - {0}
    top    = set(np.unique(lw[0, :])) - {0}
    bottom = set(np.unique(lw[-1, :])) - {0}
    return (len(left & right) > 0) or (len(top & bottom) > 0)


#Plot cluster size dist

fig, axes = plt.subplots(1, len(q_list), figsize=(18, 8), sharey=True)
if len(q_list) == 1:
    axes = [axes]

for ax, q in zip(axes, q_list):
    for L in L_list:
        allarea = []

        for _ in range(M):
            # use of mask for impermeables
            imp = (np.random.rand(L, L) < q)        

            #Can only traverse sites which have not been marked as closed
            occ = (np.random.rand(L, L) < p)
            m = occ & (~imp)                        

            lw, num = label(m, structure=structure4)

            
            if num > 0:
                labels = np.arange(1, num + 1)
                area = ndi_sum(m, lw, index=labels)
                allarea.extend(area.tolist())

        allarea = np.array(allarea, dtype=float)
        if allarea.size == 0:
            continue

        
        logamax = np.ceil(np.log(allarea.max()) / np.log(a))
        logamax = max(1, int(logamax))
        logbins = a ** np.arange(0, logamax + 1)

        nl, _ = np.histogram(allarea, bins=logbins)
        ds = np.diff(logbins)
        sl = 0.5 * (logbins[1:] + logbins[:-1])

        nsl = nl / (M * L**2 * ds)   

        col, ls = styles[L]
        ax.loglog(sl[nl > 0], nsl[nl > 0], label=f"L={L}", linewidth=3, color=col, linestyle=ls)

    #ax.set_title(f"Cluster size distribution at p={p}, q={q}", fontsize=16)
    ax.set_xlabel("$s$", fontsize=25)
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=24)

axes[0].set_ylabel("$n(s,p)$", fontsize=25)
plt.tight_layout()
plt.show()

#Spanning prob v p
L_span = max(L_list)               
p_values = np.linspace(0.45, 1, 14)

M_span = 600   

fig, ax = plt.subplots(figsize=(8, 6))

for q in q_list:
    Pi = []
    for ptest in p_values:
        count_span = 0
        for _ in range(M_span):
            imp = (np.random.rand(L_span, L_span) < q)
            occ = (np.random.rand(L_span, L_span) < ptest)
            m = occ & (~imp)

            lw, num = label(m, structure=structure4)
            if num > 0 and spans(lw):
                count_span += 1

        Pi.append(count_span / M_span)

    ax.plot(p_values, Pi, marker="o", label=f"q={q}")

ax.set_xlabel("$p$", fontsize=25)
ax.set_ylabel("$\\Pi(p,L)$", fontsize=25)
ax.set_title(f"Spanning probability vs p (L={L_span})", fontsize=30)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

#Cluster fraction plot
fig, ax = plt.subplots(figsize=(8, 6))

for q in q_list:
    frac = []
    for ptest in p_values:
        smax_list = []
        for _ in range(M_span):
            imp = (np.random.rand(L_span, L_span) < q)
            occ = (np.random.rand(L_span, L_span) < ptest)
            m = occ & (~imp)

            lw, num = label(m, structure=structure4)
            if num > 0:
                labels = np.arange(1, num + 1)
                areas = ndi_sum(m, lw, index=labels)
                smax = areas.max() if areas.size else 0.0
            else:
                smax = 0.0

            smax_list.append(smax)

        frac.append(np.mean(smax_list) / (L_span**2))

    ax.plot(p_values, frac, marker="o", label=f"q={q}")

ax.set_xlabel("$p$", fontsize=25)
ax.set_ylabel("$\\langle S_{\\max}\\rangle / L^2$", fontsize=25)
ax.set_title(f"Largest cluster fraction vs p (L={L_span})", fontsize=30)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()
