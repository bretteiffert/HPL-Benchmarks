# kernels.jl — seven-point stencil device code (KernelAbstractions).
# Unchanged from the original seven-point-stencil.jl. The kernels use KA's
# default index checking, so any grid / workgroup shape is valid.

module Kernels

using KernelAbstractions

export test_function_kernel!, laplacian_kernel!

@kernel function test_function_kernel!(u, hx, hy, hz)
    i, j, k = @index(Global, NTuple)

    T = eltype(u)
    nx, ny, nz = size(u)

    c = T(0.5)
    x = T(i - 1) * hx
    y = T(j - 1) * hy
    z = T(k - 1) * hz
    Lx = T(nx) * hx
    Ly = T(ny) * hy
    Lz = T(nz) * hz

    @inbounds u[i, j, k] = c * x * (x - Lx) +
                            c * y * (y - Ly) +
                            c * z * (z - Lz)
end

@kernel function laplacian_kernel!(f, @Const(u), invhx2, invhy2, invhz2, invhxyz2)
    i, j, k = @index(Global, NTuple)

    nx, ny, nz = size(u)

    if !(i == 1 || i >= nx || j == 1 || j >= ny || k == 1 || k >= nz)
        @inbounds f[i, j, k] = u[i, j, k] * invhxyz2 +
                               (u[i - 1, j, k] + u[i + 1, j, k]) * invhx2 +
                               (u[i, j - 1, k] + u[i, j + 1, k]) * invhy2 +
                               (u[i, j, k - 1] + u[i, j, k + 1]) * invhz2
    end
end

end # module Kernels
