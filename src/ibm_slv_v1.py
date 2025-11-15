import numpy as np
import math
import matplotlib.pyplot as plt
import os
import random

# ----- Parameters -----

Lx, Ly = 18.75, 18.75
Nx, Ny = 150, 150
h = Lx/Nx

Nk = 19
M = 320
r = 1.0

sigma = 0.1
alpha = 1.0
dt = 0.05*h*h
Tt = 400*dt
nstep = int(Tt/dt)
tol = 1e-8
gap = 1.05


# ----- Eulerian grid points -----

i = np.arange(1,Nx+1)
j = np.arange(1,Ny+1)

xi = (i-0.5)*h
yj = (j-0.5)*h

# ----- Kernel -----

def phi(r):
    ar = abs(r)

    if ar < 1.0:
        return (3.0 - 2.0*ar + np.sqrt(1.0 + 4.0*ar - 4.0*ar*ar))/8.0
    elif ar <= 2.0:
        return (5.0 - 2.0*ar + np.sqrt(9.0 - 4.0*ar + 4.0*(2.0-ar)**2))/8.0
    else:
        return 0.0

phi_vec = np.vectorize(phi)

def delta2h(dx,dy,h):
    return (1.0/(h*h))*phi_vec(dx/h)*phi_vec(dy/h)

# ----- Initialization Functions -----

def hexa_ctrs(Lx, Ly, r, Nk, gap=1.05):
    sp = 2*r + gap
    
    dx = sp
    dy = sp * np.sqrt(3) / 2
    
    nx = int(math.floor((Lx - 2*r) / dx)) + 1
    ny = int(math.floor((Ly - 2*r) / dy)) + 1
    
    xs = (Lx - (nx-1)*dx) / 2
    ys = (Ly - (ny-1)*dy) / 2
    
    ctrs = []
    for j in range(ny):
        y = ys + j * dy
        
        xoff = (dx / 2) if (j % 2 == 1) else 0
        
        for i in range(nx):
            x = xs + i * dx + xoff
            
            if (x - r >= 0 and x + r <= Lx and 
                y - r >= 0 and y + r <= Ly):
                ctrs.append([x, y])
    
    ctrs = np.array(ctrs)

    if Nk > len(ctrs):
        print(f"Warning: Requested Nk={Nk} but only {len(ctrs)} circles fit. Using all {len(ctrs)}.")
    else:
        cx, cy = Lx/2, Ly/2
        dists = np.sqrt((ctrs[:, 0] - cx)**2 + (ctrs[:, 1] - cy)**2)
        idx = np.argsort(dists)[:Nk]
        ctrs = ctrs[idx]

    return ctrs


def sqr_ctrs(Lx, Ly, r, Nk, gap=0.2):

    sp = 2*r+gap
    
    nx = int(math.floor((Lx - 2*r)/sp))+1
    ny = int(math.floor((Ly - 2*r)/sp))+1

    xs = (Lx - (nx-1)*sp)/2 + np.arange(nx) * sp
    ys = (Ly - (ny-1)*sp)/2 + np.arange(ny) * sp

    ctrs = []
    for j in ys:
        for i in xs:
            ctrs.append([i,j])

    ctrs = np.array(ctrs)

    if Nk > len(ctrs):
        print(f"Warning: Requested Nk={Nk} but only {len(ctrs)} circles fit. Using all {len(ctrs)}.")
    else:
        cx, cy = Lx/2, Ly/2
        dists = np.sqrt((ctrs[:, 0] - cx)**2 + (ctrs[:, 1] - cy)**2)
        idx = np.argsort(dists)[:Nk]
        ctrs = ctrs[idx]
    
    assert np.all(ctrs[:, 0] >= r) and np.all(ctrs[:, 0] <= Lx - r)
    assert np.all(ctrs[:, 1] >= r) and np.all(ctrs[:, 1] <= Ly - r)    

    return ctrs
    

def polar_ctrs(Lx, Ly, r, Nk, gap=0.3, max_r=None):

    cx, cy = Lx / 2, Ly / 2
    ctrs = [[cx, cy]]
    
    sp = 2*r + gap

    if max_r is None:
        max_r = min(cx, cy) - r
    
    ring_id = 1
    while True:
        if Nk is not None and len(ctrs) >= Nk:
            break
            
        R = ring_id * sp
        
        if R > max_r:
            break
        
        circ = 2 * np.pi * R
        n_ring = max(1, int(np.floor(circ / sp)))

        if Nk is not None:
            n_ring = min(n_ring, Nk - len(ctrs))
        
        ang = np.linspace(0, 2*np.pi, n_ring, endpoint=False)
        
        for tht in ang:
            x = cx + R * np.cos(tht)
            y = cy + R * np.sin(tht)
            
            if (x - r >= 0 and x + r <= Lx and 
                y - r >= 0 and y + r <= Ly):
                ctrs.append([x, y])

                if Nk is not None and len(ctrs) >= Nk:
                    break
                
        ring_id += 1
        
    ctrs = np.array(ctrs)

    if Nk is not None and len(ctrs) > Nk:
        ctrs = ctrs[:Nk]
        
    return ctrs

def random_ctrs(Lx, Ly, Nk, r, max_attempts=10000):

    ctrs = []
    min_dist = 2 * r
    
    for _ in range(Nk):
        placed = False
        
        for attempt in range(max_attempts):
            x = r + np.random.rand() * (Lx - 2*r)
            y = r + np.random.rand() * (Ly - 2*r)

            valid = True
            for cx, cy in ctrs:
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < min_dist:
                    valid = False
                    break
            
            if valid:
                ctrs.append([x, y])
                placed = True
                break
        
        if not placed:
            print(f"Warning: Could only place {len(ctrs)}/{Nk} circles")
            break
    
    return np.array(ctrs)


def poisson_ctrs(Lx, Ly, r, gap=0.2, k=30):

    min_dist = 2*r + gap
    cell_size = min_dist / np.sqrt(2)

    grid_w = int(math.ceil(Lx / cell_size))
    grid_h = int(math.ceil(Ly / cell_size))
    grid = [[None for _ in range(grid_w)] for _ in range(grid_h)]
    
    def grid_coords(x, y):
        return int(x / cell_size), int(y / cell_size)
    
    def is_valid(x, y):
        if x - r < 0 or x + r > Lx or y - r < 0 or y + r > Ly:
            return False
        
        gx, gy = grid_coords(x, y)
        
        for i in range(max(0, gx-2), min(grid_w, gx+3)):
            for j in range(max(0, gy-2), min(grid_h, gy+3)):
                if grid[j][i] is not None:
                    px, py = grid[j][i]
                    if np.sqrt((x-px)**2 + (y-py)**2) < min_dist:
                        return False
        return True
    
    ctrs = []
    active = []
    
    x0 = r + np.random.rand() * (Lx - 2*r)
    y0 = r + np.random.rand() * (Ly - 2*r)
    ctrs.append([x0, y0])
    active.append([x0, y0])
    gx, gy = grid_coords(x0, y0)
    grid[gy][gx] = [x0, y0]
    
    while active:
        idx = np.random.randint(len(active))
        px, py = active[idx]
        found = False
        
        for _ in range(k):
            angle = 2 * np.pi * np.random.rand()
            radius = min_dist + np.random.rand() * min_dist
            x = px + radius * np.cos(angle)
            y = py + radius * np.sin(angle)
            
            if is_valid(x, y):
                ctrs.append([x, y])
                active.append([x, y])
                gx, gy = grid_coords(x, y)
                grid[gy][gx] = [x, y]
                found = True
                break
        
        if not found:
            active.pop(idx)
    
    return np.array(ctrs)

def cc(ctrs, r, M):

    Nk = ctrs.shape[0]
    th = np.linspace(0.0, 2.0*np.pi, M, endpoint=False)
    XL = np.zeros((Nk,M,2),dtype=float)
    for k in range(Nk):
        cx, cy = ctrs[k]
        XL[k,:,0] = cx + r * np.cos(th)
        XL[k,:,1] = cy + r * np.sin(th)
        
    return XL


# ----- Choose initialization pattern -----

# ctrs = sqr_ctrs(Lx, Ly, r, Nk, gap=0.2)
# ctrs = hexa_ctrs(Lx, Ly, r, Nk, gap=1.05)
ctrs = polar_ctrs(Lx, Ly, r, Nk, gap=0.3)
# ctrs = random_ctrs(Lx, Ly, Nk, r, max_attempts=10000)
#ctrs = poisson_ctrs(Lx, Ly, r, gap=0.2)

XL = cc(ctrs, r, M)

# ----- Boundaries -----

def wrap(pos, Lx, Ly):

    pos = np.array(pos)
    pos[...,0] = pos[...,0]%Lx
    pos[...,1] = pos[...,1]%Ly

    return pos

# ----- Arc length -----

def compt_ds(XL):

    Nk, M, _ = XL.shape
    ds = np.zeros((Nk, M),dtype=float)
    for k in range(Nk):
        for l in range(M):
            l_prev = (l-1)%M
            dx = XL[k,l,0] - XL[k,l_prev,0]
            dy = XL[k,l,1] - XL[k,l_prev,1]
            ds[k,l] = math.sqrt(dx*dx+dy*dy)

    return ds

# ----- Tangents and normals -----

def tan_norm(XL, ds):

    Nk, M, _ = XL.shape
    T = np.zeros((Nk,M,2),dtype=float)
    Nvec = np.zeros((Nk,M,2),dtype=float)

    for k in range(Nk):
        for l in range(M):
            l_prev = (l-1)%M
            l_next = (l+1)%M

            x_prev, y_prev = XL[k,l_prev]
            x_curr, y_curr = XL[k,l]
            x_next, y_next = XL[k,l_next]

            s1 = ds[k,l]
            s2 = ds[k,l_next]

            denom = s1*s2*(s1+s2)
            Tx = (s1*s1*(x_next-x_curr) + s2*s2*(x_curr-x_prev))/denom
            Ty = (s1*s1*(y_next-y_curr) + s2*s2*(y_curr-y_prev))/denom

            T[k,l,0] = Tx
            T[k,l,1] = Ty

            mag = math.hypot(Tx,Ty)

            nx = Ty/mag
            ny = -Tx/mag
            Nvec[k,l,0] = nx
            Nvec[k,l,1] = ny

    return T, Nvec

# ----- Lagrangian forces to Eulerian grid -----

def sprd_frc(XL, f, ds, xi, yj, h):

    Nk, M, _ = XL.shape
    Fx = np.zeros((Ny,Nx),dtype=float)
    Fy = np.zeros((Ny,Nx),dtype=float)

    for k in range(Nk):
        for l in range(M):
            xl = XL[k,l,0]
            yl = XL[k,l,1]
            fx = f[k,l,0]
            fy = f[k,l,1]

            l_next = (l+1)%M
            w = 0.5*(ds[k,l]+ds[k,l_next])

            xmin = xl-2.0*h
            xmax = xl+2.0*h
            ymin = yl-2.0*h
            ymax = yl+2.0*h

            ix_min = int(math.floor((xmin)/h + 0.5))
            ix_max = int(math.ceil((xmax)/h + 0.5))
            jy_min = int(math.floor((ymin)/h + 0.5))
            jy_max = int(math.ceil((ymax)/h + 0.5))

            for ix in range(ix_min, ix_max+1):
                ix_wrapped = ix%Nx
                xg = xi[ix_wrapped]

                dx = xg - xl

                if dx>Lx/2:
                    dx -= Lx
                elif dx<=-Lx/2:
                    dx += Lx

                if abs(dx)>2.0*h:
                    continue

                for jy in range(jy_min,jy_max+1):
                    jy_wrapped = jy%Ny
                    yg = yj[jy_wrapped]
                    dy = yg - yl

                    if dy > Ly/2:
                        dy -= Ly
                    elif dy <= -Ly/2:
                        dy += Ly
                    if abs(dy) > 2.0*h:
                        continue

                    kernel = delta2h(dx,dy,h)
                    Fx[jy_wrapped, ix_wrapped] += fx*kernel*w
                    Fy[jy_wrapped, ix_wrapped] += fy*kernel*w

    return Fx, Fy

# ----- Interpolating Eulerian force -----

def inter_grid(Fx, Fy, XL, xi, yj, h):

    Nk, M, _ = XL.shape
    F_pts = np.zeros((Nk,M,2),dtype=float)

    for k in range(Nk):
        for l in range(M):
            xl = XL[k,l,0]
            yl = XL[k,l,1]

            xmin = xl - 2.0*h
            xmax = xl + 2.0*h
            ymin = yl - 2.0*h
            ymax = yl + 2.0*h

            ix_min = int(math.floor((xmin)/h+0.5))
            ix_max = int(math.ceil((xmax)/h+0.5))
            jy_min = int(math.floor((ymin)/h+0.5))
            jy_max = int(math.ceil((ymax)/h+0.5))

            sum_fx = 0.0
            sum_fy = 0.0

            for ix in range(ix_min, ix_max+1):
                ix_wrapped = ix%Nx
                xg = xi[ix_wrapped]

                dx = xg-xl
                if dx >Lx/2:
                    dx -= Lx
                elif dx <= -Lx/2:
                    dx += Lx
                if abs(dx) > 2.0*h:
                    continue

                for jy in range(jy_min, jy_max+1):
                    jy_wrapped = jy%Ny
                    yg = yj[jy_wrapped]
                    dy = yg-yl

                    if dy > Ly/2:
                        dy -= Ly
                    elif dy <= -Ly/2:
                        dy += Ly
                    if abs(dy) > 2.0*h:
                        continue

                    kernel = delta2h(dx,dy,h)
                    sum_fx += Fx[jy_wrapped, ix_wrapped] * kernel
                    sum_fy += Fy[jy_wrapped, ix_wrapped] * kernel

            F_pts[k,l,0] = sum_fx * (h*h)
            F_pts[k,l,1] = sum_fy * (h*h)

    return F_pts

# ----- Visualization Functions -----

def plt_pattern(Lx, Ly, r, ctrs, name, save_name):
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_aspect('equal')
    ax.set_title(f"{name} (N={len(ctrs)})", fontsize=14, fontweight='bold')
    ax.set_facecolor('maroon')
    ax.set_xlabel(r'$x$', fontsize=16)
    ax.set_ylabel(r'$y$', fontsize=16)
    
    for x, y in ctrs:
        circle = plt.Circle((x, y), r, color='orange', fill=False, linewidth=2)
        ax.add_patch(circle)
    
    min_dist = 2*r
    overlap_count = 0
    for i in range(len(ctrs)):
        for j in range(i+1, len(ctrs)):
            dist = np.sqrt((ctrs[i,0] - ctrs[j,0])**2 + (ctrs[i,1] - ctrs[j,1])**2)
            if dist < min_dist - 1e-10:
                ax.plot([ctrs[i,0], ctrs[j,0]], [ctrs[i,1], ctrs[j,1]], 
                       'r-', linewidth=3, alpha=0.8)
                overlap_count += 1
        
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, facecolor='white')
    print(f"  Saved: {save_name}")
    plt.show()

def vpatt(Lx, Ly, r, Nk, gap):
    
    print(f"Domain: {Lx} x {Ly}")
    print(f"Radius: {r}")
    print(f"Target Nk: {Nk}")
    print(f"Gap: {gap}\n")
    
    os.makedirs("pattern_plots", exist_ok=True)
    
    patterns = [
        ("Square", sqr_ctrs(Lx, Ly, r, Nk, gap), "pattern_plots/01_square.png"),
        ("Hexagonal", hexa_ctrs(Lx, Ly, r, Nk, gap), "pattern_plots/02_hexagonal.png"),
        ("Polar", polar_ctrs(Lx, Ly, r, Nk, gap), "pattern_plots/03_polar.png"),
        ("Random", random_ctrs(Lx, Ly, Nk, r), "pattern_plots/04_random.png"),
        ("Poisson Disk", poisson_ctrs(Lx, Ly, r, gap), "pattern_plots/05_poisson.png"),
    ]
    
    for name, ctrs, save_name in patterns:
        print(f"\n{name}:")
        print(f"  Circles placed: {len(ctrs)}")
        
        if len(ctrs) > 0:
            try:
                assert np.all(ctrs[:, 0] >= r), "circles extend past left boundary"
                assert np.all(ctrs[:, 0] <= Lx-r), "circles extend past right boundary"
                assert np.all(ctrs[:, 1] >= r), "circles extend past bottom boundary"
                assert np.all(ctrs[:, 1] <= Ly-r), "circles extend past top boundary"
                print("All circles within bounds")
                
                min_dist = 2*r
                overlap_count = 0
                for i in range(len(ctrs)):
                    for j in range(i+1, len(ctrs)):
                        dist = np.sqrt((ctrs[i,0] - ctrs[j,0])**2 + (ctrs[i,1] - ctrs[j,1])**2)
                        if dist < min_dist - 1e-10:
                            overlap_count += 1
                
                if overlap_count > 0:
                    print(f" WARNING: {overlap_count} overlaps detected!")
                else:
                    print(" No overlaps")
                    
            except AssertionError as e:
                print(f" ERROR: {e}")
        
        plt_pattern(Lx, Ly, r, ctrs, name, save_name)

def init_plot(Lx, Ly, outdir="frames"):
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_facecolor('maroon')
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$', fontsize=16)
    ax.set_ylabel(r'$y$', fontsize=16)

    os.makedirs(outdir, exist_ok=True)

    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       verticalalignment='top', fontsize=16,
                       color='white', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    return fig, ax, [], 0, time_text

def upd_plt(fig, ax, lines, XL, frame_id, curt, outdir="frames", time_text=None):

    Nk, M, _ = XL.shape

    if len(lines) == 0:
        for k in range(Nk):
            (line,) = ax.plot([], [], '-', lw=2, color='orange')
            lines.append(line)

    for k in range(Nk):
        x = XL[k,:,0]
        y = XL[k,:,1]
        lines[k].set_data(x, y)
        
    if time_text is not None:
        time_text.set_text(f'Time: {curt:.4f}')
        
    fig.canvas.draw()

    fname = f"{outdir}/fr_{frame_id:04d}.png"
    fig.savefig(fname, dpi=150)

    plt.pause(0.001)

    return frame_id + 1

def close_plot(fig):

    plt.close(fig)
    
# ----- Main Simulation Function -----

def run(XL, xi, yj, h, dt, nstep, alpha, sigma, tol):

    Nk, M, _ = XL.shape
    hist = {
        'max_force': [],
        'time': [],
    }

    vpatt(Lx, Ly, r, Nk, gap)
    
    print("\n" + "="*60)
    print("Starting simulation")
    print("="*60)
    print(f"Using pattern with {Nk} circles")
    print(f"Time steps: {nstep}")
    print(f"dt: {dt:.6f}")
    print(f"Total time: {Tt:.4f}")
    print("="*60 + "\n")
    
    fig, ax, lines, frame_id, time_text = init_plot(Lx, Ly)
    frame_id = upd_plt(fig, ax, lines, XL, frame_id, 0.0, time_text=time_text)

    for step in range(nstep):
        curt = step * dt
        ds = compt_ds(XL)
        T, Nvec = tan_norm(XL, ds)
        f = sigma * Nvec
        
        Fx, Fy = sprd_frc(XL, f, ds, xi, yj, h)

        magF = np.sqrt(Fx*Fx + Fy*Fy)
        maxF = np.max(magF)
        hist['max_force'].append(maxF)
        hist['time'].append(step*dt)

        if maxF < tol:
            print(f"\n[step {step}] Equilibrium reached: max |F| = {maxF:.5e} < tol = {tol}")
            break

        F_pts = inter_grid(Fx, Fy, XL, xi, yj, h)

        for k in range(Nk):
            for l in range(M):
                XL[k, l, 0] += dt*alpha*F_pts[k, l, 0]
                XL[k, l, 1] += dt*alpha*F_pts[k, l, 1]
        
        XL = wrap(XL, Lx, Ly)

        if step % 2 == 0:
            print(f"step {step:5d} / {nstep}  time = {curt:.4f}  max|F| = {maxF:.5e}")
            frame_id = upd_plt(fig, ax, lines, XL, frame_id, curt, time_text=time_text)

    close_plot(fig)
    print("\n" + "="*60)
    print("Simulation Complete!")
    print("="*60 + "\n")
    
    return XL, history

# ----- Run Simulation -----

print("\n" + "="*70)
print("IBM PATTERN FORMATION SIMULATION")
print("="*70)

Xg, Yg = np.meshgrid(xi, yj, indexing='xy')
XL_f, hist = run(XL, xi, yj, h, dt, nstep, alpha, sigma, tol)

# ----- Final Results Plots -----

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, state, title in zip(axes, [XL, XL_f], ["Initial Configuration", "Final Configuration"]):
    ax.set_facecolor('maroon')
    ax.scatter(Xg, Yg, s=0.6, color='lightgray', alpha=0.5)
    for k in range(state.shape[0]):
        ax.plot(state[k,:,0], state[k,:,1], color='orange', linewidth=2)
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(r'$x$', fontsize=16)
    ax.set_ylabel(r'$y$', fontsize=16)
    ax.grid(True, alpha=0.3, color='white')

plt.tight_layout()
plt.savefig('initial_vs_final.png', dpi=150, facecolor='white')
print("Saved: initial_vs_final.png")
plt.show()

fig = plt.figure(figsize=(10, 5))
plt.plot(hist['time'], hist['max_force'], linewidth=2, color='blue')
plt.yscale('log')
plt.xlabel('Time', fontsize=12)
plt.ylabel('max |F|', fontsize=12)
plt.title('Force Evolution', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('force_evolution.png', dpi=150, facecolor='white')
print("Saved: force_evolution.png")
plt.show()

print("\n" + "="*70)
print("ALL DONE!")
print("="*70)
