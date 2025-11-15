import numpy as np
import math
import matplotlib.pyplot as plt
import os

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
Tt = 350*dt
nstep = int(Tt/dt)
tol = 1e-8
sp = 1.1

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

# ----- Hexagonal centers -----

def hexa_ctrs(Lx,Ly,r,rings,sp=1.05):

    d = 2*r*sp
    ctrs = [(0.0,0.0)]

    for n in range(1,rings+1):
        for kdir in range(6):
            ang = np.pi/3 * kdir
            xs = n*d*np.cos(ang)
            ys = n*d*np.sin(ang)
            for jstep in range(n):
                x = xs - jstep*d*np.cos(ang-np.pi/3)
                y = ys - jstep*d*np.sin(ang-np.pi/3)
                ctrs.append((x,y))
    ctrs = np.array(ctrs)

    ctrs[:,0] += Lx/2
    ctrs[:,1] += Ly/2

    return ctrs

def rings_Nk(Nk):
    n=0
    while 1+3*n*(n+1)<Nk:
        n += 1
    return n

rings = rings_Nk(Nk)
ctrs = hexa_ctrs(Lx,Ly,r,rings,sp)
ctrs = ctrs[:Nk]

# ----- Initialization of lagrangian boundaries -----

tht = np.linspace(0.0, 2.0*np.pi, M, endpoint=False)
XL = np.zeros((len(ctrs),M,2),dtype=float)
for k in range(len(ctrs)):
    cxk, cyk = ctrs[k]
    XL[k,:,0] = cxk+r*np.cos(tht)
    XL[k,:,1] = cyk+r*np.sin(tht)

Nk_actual = XL.shape[0]

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

            ds[k,l] = np.sqrt(dx*dx+dy*dy)

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

# ----- Dyanmic plot -----

def init_plot(Lx, Ly, outdir="frames"):

    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_facecolor('maroon')
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_aspect('equal')

    os.makedirs(outdir, exist_ok=True)

    return fig, ax, [], 0

def upd_plt(fig, ax, lines, XL, frame_id, outdir="frames"):

    Nk, M, _ = XL.shape

    if len(lines) == 0:
        for k in range(Nk):
            (line,) = ax.plot([], [], '-', lw=2, color='orange')
            lines.append(line)

    for k in range(Nk):
        x = XL[k,:,0]
        y = XL[k,:,1]
        lines[k].set_data(x, y)

    fig.canvas.draw()

    fname = f"{outdir}/frame_{frame_id:04d}.png"
    fig.savefig(fname, dpi=300)

    plt.pause(0.001)
    
    return frame_id + 1

def close_plot(fig):

    plt.close(fig)

# ----- Time loop -----

def run(XL, xi, yj, h, dt, nstep, alpha, sigma, tol):

    Nk, M, _ = XL.shape
    hist = {
        'max_force': [],
        'time': [],
        }

    fig, ax, lines, frame_id = init_plot(Lx, Ly)
    frame_id = upd_plt(fig, ax, lines, XL, frame_id)

    for step in range(nstep):
        ds = compt_ds(XL)
        T, Nvec = tan_norm(XL,ds)
        f = sigma * Nvec
        Fx, Fy = sprd_frc(XL,f,ds,xi,yj,h)

        magF = np.sqrt(Fx*Fx+Fy*Fy)
        maxF = np.max(magF)
        hist['max_force'].append(maxF)
        hist['time'].append(step*dt)

        if maxF < tol:
            print(f"[step {step}] Equilibrium reached: max |F| = {maxF:.5e} < tol = {tol}")
            break

        F_pts = inter_grid(Fx, Fy, XL, xi, yj, h)

        for k in range(Nk):
            for l in range(M):
                XL[k,l,0] += dt*alpha*F_pts[k,l,0]
                XL[k,l,1] += dt*alpha*F_pts[k,l,1]
        XL = wrap(XL, Lx, Ly)

        if step % 2 == 0:
            print(f"step {step:5d} max|F| = {maxF:.5e}")
            frame_id = update_plot(fig, ax, lines, XL, frame_id)

    close_plot(fig)
    
    return XL, history

print("------------------")
print("Running simulation")
print("------------------")

Xg, Yg = np.meshgrid(xi,yj, indexing='xy')
XL_f, hist = run(XL,xi,yj, h, dt, nstep, alpha, sigma, tol)

print("Simulation completed!")

fig, axes = plt.subplots(1,2,figsize=(12,6))
for ax, state, title in zip(axes, [XL, XL_f], ["Initial", "Final"]):
    ax.set_facecolor('maroon')
    ax.scatter(Xg,Yg, s=0.6, color='lightgray')
    for k in range(state.shape[0]):
        ax.plot(state[k,:,0],state[k,:,1], color='orange')
        ax.set_xlim(0,Lx)
        ax.set_ylim(0,Ly)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel(r'$x$',fontsize=16)
        ax.set_ylabel(r'$y$',fontsize=16)
plt.show()

plt.figure(figsize=(6,3))
plt.plot(hist['time'],hist['max_force'])
plt.yscale('log')
plt.xlabel('time')
plt.ylabel(r'max $|F|$')
plt.title('Evolution of force')
plt.grid(True)
plt.show()
