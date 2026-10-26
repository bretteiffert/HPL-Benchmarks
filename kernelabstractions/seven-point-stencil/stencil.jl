# stencil.jl — grid buffers, timed launches, theoretical traffic and the
# on-device correctness check (∇²u = 3 at interior points). No printing here.

module Stencil

using KernelAbstractions
using ..Kernels
using ..Backend: elapsed_seconds, device_sync

export StencilDevice, init_field!, laplacian!, timed_laplacian!, traffic_bytes,
       check_interior, StencilCheck

struct StencilDevice{T,B,A,K1,K2}
    backend::B
    u::A
    f::A
    nx::Int
    ny::Int
    nz::Int
    hx::T
    hy::T
    hz::T
    invhx2::T
    invhy2::T
    invhz2::T
    invhxyz2::T
    init_kern::K1
    lap_kern::K2
end

function StencilDevice{T}(backend, nx::Int, ny::Int, nz::Int, block::NTuple{3,Int}) where {T}
    u = KernelAbstractions.allocate(backend, T, nx, ny, nz)
    f = KernelAbstractions.allocate(backend, T, nx, ny, nz)
    hx = T(1) / T(nx - 1)
    hy = T(1) / T(ny - 1)
    hz = T(1) / T(nz - 1)
    invhx2 = one(T) / hx / hx
    invhy2 = one(T) / hy / hy
    invhz2 = one(T) / hz / hz
    invhxyz2 = T(-2) * (invhx2 + invhy2 + invhz2)
    ik = test_function_kernel!(backend, block)
    lk = laplacian_kernel!(backend, block)
    return StencilDevice{T,typeof(backend),typeof(u),typeof(ik),typeof(lk)}(
        backend, u, f, nx, ny, nz, hx, hy, hz, invhx2, invhy2, invhz2, invhxyz2, ik, lk)
end

init_field!(s::StencilDevice) =
    s.init_kern(s.u, s.hx, s.hy, s.hz; ndrange = (s.nx, s.ny, s.nz))

laplacian!(s::StencilDevice) =
    s.lap_kern(s.f, s.u, s.invhx2, s.invhy2, s.invhz2, s.invhxyz2;
               ndrange = (s.nx, s.ny, s.nz))

"One timed Laplacian (seconds, device events)."
timed_laplacian!(s::StencilDevice) = elapsed_seconds(() -> laplacian!(s))

"(fetch, write) bytes per iteration: all points minus 8 corners and 12 edges read, interior written."
function traffic_bytes(::Type{T}, nx, ny, nz) where {T}
    interior = (nx - 2) * (ny - 2) * (nz - 2)
    fetch_points = nx * ny * nz - 8 - 4 * (nx - 2) - 4 * (ny - 2) - 4 * (nz - 2)
    return fetch_points * sizeof(T), interior * sizeof(T)
end

struct StencilCheck
    max_abs_err::Float64
    mean_abs_err::Float64
    l2_rel_err::Float64
    num_checked::Int
end

"Compare the interior of f with 3 on the device (no copy of the grid to the host)."
function check_interior(s::StencilDevice)
    v = view(s.f, 2:(s.nx - 1), 2:(s.ny - 1), 2:(s.nz - 1))
    n = length(v)
    n == 0 && return StencilCheck(0.0, 0.0, 0.0, 0)
    err(x) = abs(Float64(x) - 3.0)
    max_err = mapreduce(err, max, v)
    sum_abs = mapreduce(err, +, v)
    sum_sq = mapreduce(x -> err(x)^2, +, v)
    device_sync()
    return StencilCheck(max_err, sum_abs / n, sqrt(sum_sq / (9.0 * n)), n)
end

end # module Stencil
