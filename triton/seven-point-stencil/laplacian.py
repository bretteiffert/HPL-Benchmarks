#!/usr/bin/env python3
# Triton seven-point Laplacian benchmark with configurable program tile dimensions and device-event timing.
# It uses the CUDA namespace exposed by either CUDA or ROCm PyTorch and defaults to float32 precision.
# Correctness is measured against the analytical Laplacian of the initialized quadratic field.

import sys

import numpy as np
import torch
import triton
import triton.language as tl

precision = torch.float32
precision_str = "float"

TBSize = 1024
L = 1024
NUM_ITER = 1000
csv_output = False


def resolve_device(device_arg):
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA/ROCm device visible to PyTorch.")

    idx = device_arg if device_arg is not None else torch.cuda.current_device()
    device = torch.device("cuda", idx)

    backend_label = "HIP" if torch.version.hip is not None else "CUDA"

    return device, backend_label


@triton.jit
def test_function_kernel(u_ptr, h_ptr, nx, ny, nz, TB: tl.constexpr):
    i = tl.program_id(0) * TB + tl.arange(0, TB)
    j = tl.program_id(1)
    k = tl.program_id(2)

    mask = i < nx

    hx = tl.load(h_ptr + 0)
    hy = tl.load(h_ptr + 1)
    hz = tl.load(h_ptr + 2)

    pos = (i + nx * (j + ny * k)).to(tl.int64)

    c = 0.5
    x = i.to(hx.dtype) * hx
    y = j.to(hy.dtype) * hy
    z = k.to(hz.dtype) * hz
    Lx = nx.to(hx.dtype) * hx
    Ly = ny.to(hy.dtype) * hy
    Lz = nz.to(hz.dtype) * hz

    val = c * x * (x - Lx) + c * y * (y - Ly) + c * z * (z - Lz)
    tl.store(u_ptr + pos, val, mask=mask)


@triton.jit
def laplacian_kernel(f_ptr, u_ptr, p_ptr, nx, ny, nz,
                     BLK_X: tl.constexpr, BLK_Y: tl.constexpr, BLK_Z: tl.constexpr):
    i = tl.program_id(0) * BLK_X + tl.arange(0, BLK_X)[:, None, None]
    j = tl.program_id(1) * BLK_Y + tl.arange(0, BLK_Y)[None, :, None]
    k = tl.program_id(2) * BLK_Z + tl.arange(0, BLK_Z)[None, None, :]

    interior = ((i > 0) & (i < nx - 1) &
                (j > 0) & (j < ny - 1) &
                (k > 0) & (k < nz - 1))

    invhx2 = tl.load(p_ptr + 0)
    invhy2 = tl.load(p_ptr + 1)
    invhz2 = tl.load(p_ptr + 2)
    invhxyz2 = tl.load(p_ptr + 3)

    slc = nx * ny
    pos = (i + nx * j + slc * k).to(tl.int64)

    cc = tl.load(u_ptr + pos, mask=interior, other=0.0)
    xm = tl.load(u_ptr + pos - 1, mask=interior, other=0.0)
    xp = tl.load(u_ptr + pos + 1, mask=interior, other=0.0)
    ym = tl.load(u_ptr + pos - nx, mask=interior, other=0.0)
    yp = tl.load(u_ptr + pos + nx, mask=interior, other=0.0)
    zm = tl.load(u_ptr + pos - slc, mask=interior, other=0.0)
    zp = tl.load(u_ptr + pos + slc, mask=interior, other=0.0)

    val = (cc * invhxyz2
           + (xm + xp) * invhx2
           + (ym + yp) * invhy2
           + (zm + zp) * invhz2)

    tl.store(f_ptr + pos, val, mask=interior)


def test_function(d_u, d_h, nx, ny, nz):
    grid = (triton.cdiv(nx, TBSize), ny, nz)
    test_function_kernel[grid](d_u, d_h, nx, ny, nz,
                               TB=TBSize)


def laplacian(d_f, d_u, d_p, nx, ny, nz, BLK_X, BLK_Y, BLK_Z):
    grid = (triton.cdiv(nx, BLK_X), triton.cdiv(ny, BLK_Y), triton.cdiv(nz, BLK_Z))
    laplacian_kernel[grid](d_f, d_u, d_p, nx, ny, nz,
                           BLK_X=BLK_X, BLK_Y=BLK_Y, BLK_Z=BLK_Z)


def check_correctness(d_f, nx, ny, nz):
    expected = 3.0
    sum_abs = 0.0
    sum_sq_err = 0.0
    sum_sq_exact = 0.0
    max_err = 0.0
    count = 0

    view = d_f.view(nz, ny, nx)

    for k in range(1, nz - 1):
        slab = view[k, 1:ny - 1, 1:nx - 1].to(torch.float64).cpu().numpy()
        err = np.abs(slab - expected)
        sum_abs += float(err.sum(dtype=np.float64))
        sum_sq_err += float((err * err).sum(dtype=np.float64))
        m = float(err.max())
        if m > max_err:
            max_err = m
        count += err.size

    sum_sq_exact = expected * expected * count

    if count > 0:
        mean_abs_err = sum_abs / count
        l2_rel_err = ((sum_sq_err / sum_sq_exact) ** 0.5 if sum_sq_exact > 0.0
                      else sum_sq_err ** 0.5)
    else:
        max_err = mean_abs_err = l2_rel_err = 0.0

    return dict(max_abs_err=max_err, l2_rel_err=l2_rel_err,
                mean_abs_err=mean_abs_err, num_checked=count)


def g(x):
    return f"{x:g}"


def main(argv):
    global csv_output

    BLK_X, BLK_Y, BLK_Z = TBSize, 1, 1

    nx = ny = nz = L
    device_arg = None

    if len(argv) > 1:
        nx = int(argv[1])
    if len(argv) > 2:
        ny = int(argv[2])
    if len(argv) > 3:
        nz = int(argv[3])
    if len(argv) > 4:
        BLK_X = int(argv[4])
    if len(argv) > 5:
        BLK_Y = int(argv[5])
    if len(argv) > 6:
        BLK_Z = int(argv[6])
    if len(argv) > 7 and argv[7] == "--csv":
        csv_output = True

    if "--device" in argv:
        device_arg = int(argv[argv.index("--device") + 1])

    itemsize = torch.tensor([], dtype=precision).element_size()

    theoretical_fetch_size = (nx * ny * nz - 8
                              - 4 * (nx - 2) - 4 * (ny - 2) - 4 * (nz - 2)) * itemsize
    theoretical_write_size = ((nx - 2) * (ny - 2) * (nz - 2)) * itemsize
    datasize = theoretical_fetch_size + theoretical_write_size

    device, backend_label = resolve_device(device_arg)
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)

    if csv_output:
        print("backend,GPU,precision,L,blk_x,blk_y,blk_z,BW_GBs")
    else:
        print(gpu_name)
        print("Backend:", backend_label)
        print("Precision:", precision_str)
        print(f"nx,ny,nz = {nx}, {ny}, {nz}")
        print(f"tile shape = {BLK_X}, {BLK_Y}, {BLK_Z} (num_warps chosen by Triton)")
        print("Theoretical fetch size (GB):", g(theoretical_fetch_size * 1e-9))
        print("Theoretical write size (GB):", g(theoretical_write_size * 1e-9))

    n = nx * ny * nz

    d_u = torch.empty(n, dtype=precision, device=device)
    d_f = torch.empty(n, dtype=precision, device=device)

    npdt = np.float64 if precision == torch.float64 else np.float32
    hx = npdt(1.0 / (nx - 1))
    hy = npdt(1.0 / (ny - 1))
    hz = npdt(1.0 / (nz - 1))

    d_h = torch.tensor([hx, hy, hz], dtype=precision, device=device)

    invhx2 = npdt(1) / hx / hx
    invhy2 = npdt(1) / hy / hy
    invhz2 = npdt(1) / hz / hz
    invhxyz2 = npdt(-2) * (invhx2 + invhy2 + invhz2)
    d_p = torch.tensor([invhx2, invhy2, invhz2, invhxyz2],
                       dtype=precision, device=device)

    test_function(d_u, d_h, nx, ny, nz)

    laplacian(d_f, d_u, d_p, nx, ny, nz, BLK_X, BLK_Y, BLK_Z)
    torch.cuda.synchronize()

    corr = check_correctness(d_f, nx, ny, nz)

    if csv_output:
        print("correctness,precision,max_abs_err,l2_rel_err,mean_abs_err,points_checked",
              file=sys.stderr)
        print(f"correctness,{precision_str},{g(corr['max_abs_err'])},"
              f"{g(corr['l2_rel_err'])},{g(corr['mean_abs_err'])},{corr['num_checked']}",
              file=sys.stderr)
    else:
        print(f"--- Correctness metrics (precision = {precision_str}) ---")
        print("Interior points checked :", corr["num_checked"])
        print("Max abs error           :", g(corr["max_abs_err"]))
        print("Mean abs error          :", g(corr["mean_abs_err"]))
        print("Relative L2 error       :", g(corr["l2_rel_err"]))
        print()

    total_elapsed = 0.0
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    for _ in range(NUM_ITER):
        torch.cuda.synchronize()
        start.record()
        laplacian(d_f, d_u, d_p, nx, ny, nz, BLK_X, BLK_Y, BLK_Z)
        stop.record()
        stop.synchronize()
        elapsed = start.elapsed_time(stop)
        if csv_output:
            print(f"Triton-{backend_label},{gpu_name},{precision_str},{nx},"
                  f"{BLK_X},{BLK_Y},{BLK_Z},{g(datasize / elapsed / 1e6)}")
        total_elapsed += elapsed

    if not csv_output:
        print(f"Laplacian kernel took: {g(total_elapsed / NUM_ITER)} ms, "
              f"effective memory bandwidth: "
              f"{g(datasize * NUM_ITER / total_elapsed / 1e6)} GB/s \n")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
