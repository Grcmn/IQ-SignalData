import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Parameter ──────────────────────────────────────────────────────────────────
c = 3e8  # Lichtgeschwindigkeit [m/s]

# Positionen von Sender und Empfänger [km]
sender     = np.array([0.0,   0.0])
empfaenger = np.array([80.0,  0.0])

# Flugzeug-Position [km]
ziel = np.array([60.0, 55.0])

# Zweites Empfangssystem (für Schnittpunkt-Demo)
empfaenger2 = np.array([40.0, -30.0])

# ── Bistatic Geometrie berechnen ───────────────────────────────────────────────
def bistatic_ellipse(sender, empfaenger, bistatic_range_km, n=1000):
    """
    Berechnet alle Punkte einer bistatic Ellipse.

    Ellipsen-Parameter:
      Mittelpunkt = Mitte zwischen Sender und Empfänger
      a = bistatic_range / 2          (große Halbachse)
      c = Abstand Mittelpunkt-Brennpunkt = baseline / 2
      b = sqrt(a^2 - c^2)             (kleine Halbachse)
    """
    baseline    = np.linalg.norm(empfaenger - sender)
    a           = bistatic_range_km / 2.0
    c_val       = baseline / 2.0

    if a <= c_val:
        print(f"  Warnung: bistatic_range ({bistatic_range_km:.1f} km) "
              f"muss groesser sein als Baseline ({baseline:.1f} km)!")
        return None, None

    b           = np.sqrt(a**2 - c_val**2)
    mittelpunkt = (sender + empfaenger) / 2.0

    # Winkel der Baseline (Rotation der Ellipse)
    winkel = np.arctan2(empfaenger[1] - sender[1],
                        empfaenger[0] - sender[0])

    # Ellipse im lokalen Koordinatensystem
    t  = np.linspace(0, 2 * np.pi, n)
    x0 = a * np.cos(t)
    y0 = b * np.sin(t)

    # Rotation in globales Koordinatensystem
    x = mittelpunkt[0] + x0 * np.cos(winkel) - y0 * np.sin(winkel)
    y = mittelpunkt[1] + x0 * np.sin(winkel) + y0 * np.cos(winkel)

    return x, y


def berechne_parameter(ziel, sender, empfaenger):
    """Berechnet R_T, R_R, bistatic Range und Laufzeit."""
    R_T             = np.linalg.norm(ziel - sender)
    R_R             = np.linalg.norm(ziel - empfaenger)
    bistatic_range  = R_T + R_R
    tau             = bistatic_range * 1e3 / c  # in Sekunden (km → m)
    return R_T, R_R, bistatic_range, tau


# ── Berechnung ─────────────────────────────────────────────────────────────────
R_T, R_R, br1, tau1 = berechne_parameter(ziel, sender, empfaenger)
_,   _,   br2, tau2 = berechne_parameter(ziel, sender, empfaenger2)

print("=" * 50)
print("  Bistatic Radar – Ellipsen Berechnung")
print("=" * 50)
print(f"\n  Sender:       ({sender[0]:.0f}, {sender[1]:.0f}) km")
print(f"  Empfaenger 1: ({empfaenger[0]:.0f}, {empfaenger[1]:.0f}) km")
print(f"  Empfaenger 2: ({empfaenger2[0]:.0f}, {empfaenger2[1]:.0f}) km")
print(f"  Ziel:         ({ziel[0]:.0f}, {ziel[1]:.0f}) km")
print()
print(f"  R_T  (Sender → Ziel):       {R_T:.2f} km")
print(f"  R_R  (Ziel → Empf. 1):      {R_R:.2f} km")
print(f"  Bistatic Range 1:            {br1:.2f} km")
print(f"  Laufzeit tau 1:              {tau1*1e6:.2f} µs")
print()
print(f"  Bistatic Range 2:            {br2:.2f} km")
print(f"  Laufzeit tau 2:              {tau2*1e6:.2f} µs")
print("=" * 50)

# ── Ellipsen-Punkte ────────────────────────────────────────────────────────────
x1, y1 = bistatic_ellipse(sender, empfaenger,  br1)
x2, y2 = bistatic_ellipse(sender, empfaenger2, br2)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#f7f7f5')
ax.set_facecolor('#f7f7f5')

# Ellipse 1
if x1 is not None:
    ax.plot(x1, y1, color='#378ADD', linewidth=2.0, label=f'Ellipse 1  (R_T+R_R = {br1:.1f} km)')
    ax.fill(x1, y1, alpha=0.05, color='#378ADD')

# Ellipse 2
if x2 is not None:
    ax.plot(x2, y2, color='#D85A30', linewidth=2.0, linestyle='--',
            label=f'Ellipse 2  (R_T+R_R = {br2:.1f} km)')
    ax.fill(x2, y2, alpha=0.05, color='#D85A30')

# R_T und R_R Linien zum Ziel
ax.plot([sender[0],     ziel[0]], [sender[1],     ziel[1]],
        color='#378ADD', linewidth=1.2, linestyle=':', alpha=0.7)
ax.plot([empfaenger[0], ziel[0]], [empfaenger[1], ziel[1]],
        color='#888', linewidth=1.2, linestyle=':', alpha=0.7)
ax.plot([empfaenger2[0], ziel[0]], [empfaenger2[1], ziel[1]],
        color='#D85A30', linewidth=1.2, linestyle=':', alpha=0.7)

# Beschriftungen der Linien
mid_RT  = (sender     + ziel) / 2
mid_RR  = (empfaenger + ziel) / 2
mid_RR2 = (empfaenger2 + ziel) / 2
ax.text(mid_RT[0]-4,  mid_RT[1]+1,  f'R_T = {R_T:.1f} km',
        fontsize=9, color='#378ADD', fontstyle='italic')
ax.text(mid_RR[0]+1,  mid_RR[1]+1,  f'R_R = {R_R:.1f} km',
        fontsize=9, color='#555', fontstyle='italic')

# Baseline
ax.annotate('', xy=empfaenger, xytext=sender,
            arrowprops=dict(arrowstyle='<->', color='#aaa', lw=1.2))
ax.text((sender[0]+empfaenger[0])/2, sender[1]-5,
        f'Baseline = {np.linalg.norm(empfaenger-sender):.0f} km',
        ha='center', fontsize=9, color='#888')

# Punkte: Sender, Empfänger, Ziel
ax.scatter(*sender,      s=120, color='#378ADD', zorder=5, edgecolors='white', linewidths=1.5)
ax.scatter(*empfaenger,  s=120, color='#888780', zorder=5, edgecolors='white', linewidths=1.5)
ax.scatter(*empfaenger2, s=120, color='#D85A30', zorder=5, edgecolors='white', linewidths=1.5)
ax.scatter(*ziel,        s=160, color='#1D9E75', zorder=6, edgecolors='white', linewidths=1.5, marker='^')

# Labels
offset = 3
ax.text(sender[0]-2,      sender[1]+offset+1,     'Sender (DVB-T)',  fontsize=10, fontweight='bold', color='#378ADD')
ax.text(empfaenger[0]-8,  empfaenger[1]+offset+1, 'Empfänger 1',     fontsize=10, fontweight='bold', color='#555')
ax.text(empfaenger2[0]+2, empfaenger2[1]-offset-4,'Empfänger 2',     fontsize=10, fontweight='bold', color='#D85A30')
ax.text(ziel[0]+2,        ziel[1]+offset,          '✈ Ziel',          fontsize=11, fontweight='bold', color='#1D9E75')

# Schnittpunkt-Hinweis
ax.annotate('Schnittpunkt\n= Zielposition',
            xy=ziel, xytext=(ziel[0]+18, ziel[1]-18),
            fontsize=9, color='#1D9E75',
            arrowprops=dict(arrowstyle='->', color='#1D9E75', lw=1.2))

# Info-Box
info = (f"R_T  = {R_T:.1f} km\n"
        f"R_R  = {R_R:.1f} km\n"
        f"R_T+R_R = {br1:.1f} km\n"
        f"τ₁  = {tau1*1e6:.1f} µs")
ax.text(0.02, 0.97, info, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#ccc', alpha=0.85))

# Achsen
ax.set_xlabel('x [km]', fontsize=11)
ax.set_ylabel('y [km]', fontsize=11)
ax.set_title('Bistatic Ellipsen – Passivradar Geometrie', fontsize=13, fontweight='bold', pad=14)
ax.legend(loc='lower right', fontsize=9, framealpha=0.85)
ax.grid(True, linestyle='--', alpha=0.3, color='#aaa')
ax.set_aspect('equal')
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig('bistatic_ellipse.png', dpi=150, bbox_inches='tight')
print("\n  Gespeichert: bistatic_ellipse.png")
plt.close()