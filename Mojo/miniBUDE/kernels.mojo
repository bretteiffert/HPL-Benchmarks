# miniBUDE device code (Mojo 1.2.0 / MAX 26.7): the fasten_main kernel.
#
# The kernel body is unchanged from the original miniBUDE.mojo. Changes forced
# by Mojo 1.1+:
#   * GPU index builtins come from max.gpu (std.gpu is private);
#   * `InlineArray` was removed; `Array` is the same type with the same
#     `fill=` / `uninitialized=` constructors.
# The kernel uses the native std.math sin/cos. Mojo 1.2 rejects float64 sin/cos
# on NVIDIA at compile time and has no float64 lowering on AMD, so the float64
# instantiation is only compiled when the program is built with -D FP64 (see
# main.mojo).

from max.gpu import block_dim, block_idx, thread_idx
from std.collections import Array
from std.math import cos, sin, sqrt
from std.memory import Pointer
from std.utils.numerics import max_finite

comptime HBTYPE_F: Int32 = 70
comptime HBTYPE_E: Int32 = 69

comptime PPWIS_STR = "1,2,4,8,16,32,64,128"


def is_supported_ppwi(p: Int) -> Bool:
    return p == 1 or p == 2 or p == 4 or p == 8 or p == 16 or p == 32 or p == 64 or p == 128


def fasten_main[
    dtype: DType, ppwi: Int
](
    etotals: Pointer[Scalar[dtype], MutAnyOrigin],
    prot_x: Pointer[Scalar[dtype], MutAnyOrigin],
    prot_y: Pointer[Scalar[dtype], MutAnyOrigin],
    prot_z: Pointer[Scalar[dtype], MutAnyOrigin],
    prot_t: Pointer[Int32, MutAnyOrigin],
    lig_x: Pointer[Scalar[dtype], MutAnyOrigin],
    lig_y: Pointer[Scalar[dtype], MutAnyOrigin],
    lig_z: Pointer[Scalar[dtype], MutAnyOrigin],
    lig_t: Pointer[Int32, MutAnyOrigin],
    t0: Pointer[Scalar[dtype], MutAnyOrigin],
    t1: Pointer[Scalar[dtype], MutAnyOrigin],
    t2: Pointer[Scalar[dtype], MutAnyOrigin],
    t3: Pointer[Scalar[dtype], MutAnyOrigin],
    t4: Pointer[Scalar[dtype], MutAnyOrigin],
    t5: Pointer[Scalar[dtype], MutAnyOrigin],
    ff_hbtype: Pointer[Int32, MutAnyOrigin],
    ff_radius: Pointer[Scalar[dtype], MutAnyOrigin],
    ff_hphb: Pointer[Scalar[dtype], MutAnyOrigin],
    ff_elsc: Pointer[Scalar[dtype], MutAnyOrigin],
    natlig: Int32,
    natpro: Int32,
    num_transforms: Int32,
):
    comptime assert dtype.is_floating_point(), "fasten_main requires a floating-point dtype"

    comptime ZERO = Scalar[dtype](0)
    comptime QUARTER = Scalar[dtype](0.25)
    comptime HALF = Scalar[dtype](0.5)
    comptime ONE = Scalar[dtype](1)
    comptime TWO = Scalar[dtype](2)
    comptime FOUR = Scalar[dtype](4)
    comptime CNSTNT = Scalar[dtype](45)
    comptime HARDNESS = Scalar[dtype](38)
    comptime NPNPDIST = Scalar[dtype](5.5)
    comptime NPPDIST = Scalar[dtype](1)
    comptime MAXVAL = max_finite[dtype]()

    var lsz = Int32(block_dim.x)

    var base = Int32(block_idx.x) * lsz * Int32(ppwi) + Int32(thread_idx.x)
    var ix = base if base < num_transforms else (
        num_transforms - Int32(ppwi) * lsz + Int32(thread_idx.x)
    )

    var etot = Array[Scalar[dtype], ppwi](fill=ZERO)
    var tf = Array[Scalar[dtype], ppwi * 12](uninitialized=True)
    var lpos = Array[Scalar[dtype], ppwi * 3](uninitialized=True)

    comptime for i in range(ppwi):
        var index = Int(ix + Int32(i) * lsz)

        var sx = sin(t0[unsafe_offset=index])
        var cx = cos(t0[unsafe_offset=index])
        var sy = sin(t1[unsafe_offset=index])
        var cy = cos(t1[unsafe_offset=index])
        var sz = sin(t2[unsafe_offset=index])
        var cz = cos(t2[unsafe_offset=index])

        comptime b = i * 12
        tf[b + 0] = cy * cz
        tf[b + 1] = sx * sy * cz - cx * sz
        tf[b + 2] = cx * sy * cz + sx * sz
        tf[b + 3] = t3[unsafe_offset=index]
        tf[b + 4] = cy * sz
        tf[b + 5] = sx * sy * sz + cx * cz
        tf[b + 6] = cx * sy * sz - sx * cz
        tf[b + 7] = t4[unsafe_offset=index]
        tf[b + 8] = -sy
        tf[b + 9] = sx * cy
        tf[b + 10] = cx * cy
        tf[b + 11] = t5[unsafe_offset=index]

    for il in range(Int(natlig)):
        var lx = lig_x[unsafe_offset=il]
        var ly = lig_y[unsafe_offset=il]
        var lz = lig_z[unsafe_offset=il]
        var ltype = Int(lig_t[unsafe_offset=il])

        var l_radius = ff_radius[unsafe_offset=ltype]
        var l_hphb = ff_hphb[unsafe_offset=ltype]
        var l_elsc = ff_elsc[unsafe_offset=ltype]
        var l_hbtype = ff_hbtype[unsafe_offset=ltype]

        var lhphb_ltz = l_hphb < ZERO
        var lhphb_gtz = l_hphb > ZERO

        comptime for i in range(ppwi):
            comptime b = i * 12
            comptime o = i * 3
            lpos[o + 0] = tf[b + 3] + lx * tf[b + 0] + ly * tf[b + 1] + lz * tf[b + 2]
            lpos[o + 1] = tf[b + 7] + lx * tf[b + 4] + ly * tf[b + 5] + lz * tf[b + 6]
            lpos[o + 2] = tf[b + 11] + lx * tf[b + 8] + ly * tf[b + 9] + lz * tf[b + 10]

        for ip in range(Int(natpro)):
            var px = prot_x[unsafe_offset=ip]
            var py = prot_y[unsafe_offset=ip]
            var pz = prot_z[unsafe_offset=ip]
            var ptype = Int(prot_t[unsafe_offset=ip])

            var p_radius = ff_radius[unsafe_offset=ptype]
            var p_hphb = ff_hphb[unsafe_offset=ptype]
            var p_elsc = ff_elsc[unsafe_offset=ptype]
            var p_hbtype = ff_hbtype[unsafe_offset=ptype]

            var radij = p_radius + l_radius
            var r_radij = ONE / radij

            var both_f = (p_hbtype == HBTYPE_F) and (l_hbtype == HBTYPE_F)
            var elcdst = FOUR if both_f else TWO
            var elcdst1 = QUARTER if both_f else HALF
            var type_E = (p_hbtype == HBTYPE_E) or (l_hbtype == HBTYPE_E)

            var phphb_ltz = p_hphb < ZERO
            var phphb_gtz = p_hphb > ZERO
            var phphb_nz = p_hphb != ZERO
            var p_hphb_s = p_hphb * (
                -ONE if (phphb_ltz and lhphb_gtz) else ONE
            )
            var l_hphb_s = l_hphb * (
                -ONE if (phphb_gtz and lhphb_ltz) else ONE
            )

            var distdslv: Scalar[dtype]
            if phphb_ltz:
                distdslv = NPNPDIST if lhphb_ltz else NPPDIST
            else:
                distdslv = NPPDIST if lhphb_ltz else -MAXVAL
            var r_distdslv = ONE / distdslv

            var chrg_init = l_elsc * p_elsc
            var dslv_init = p_hphb_s + l_hphb_s

            comptime for i in range(ppwi):
                comptime o = i * 3
                var x = lpos[o + 0] - px
                var y = lpos[o + 1] - py
                var z = lpos[o + 2] - pz
                var distij = sqrt(x * x + y * y + z * z)

                var distbb = distij - radij
                var zone1 = distbb < ZERO

                etot[i] += (ONE - (distij * r_radij)) * (
                    TWO * HARDNESS if zone1 else ZERO
                )

                var chrg_e = chrg_init * (
                    (ONE if zone1 else (ONE - distbb * elcdst1))
                    * (ONE if distbb < elcdst else ZERO)
                )
                var neg_chrg_e = -abs(chrg_e)
                chrg_e = neg_chrg_e if type_E else chrg_e
                etot[i] += chrg_e * CNSTNT

                var coeff = ONE - (distbb * r_distdslv)
                var dslv_e = dslv_init * (
                    ONE if (distbb < distdslv and phphb_nz) else ZERO
                )
                dslv_e *= ONE if zone1 else coeff
                etot[i] += dslv_e

    if base < num_transforms:

        comptime for i in range(ppwi):
            etotals[unsafe_offset=Int(base + Int32(i) * lsz)] = etot[i] * HALF

