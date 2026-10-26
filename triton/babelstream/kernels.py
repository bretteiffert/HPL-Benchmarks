# BabelStream device code (Triton). Kernel math is unchanged from the original
# babelstream.py; BLOCK_SIZE is a constexpr, so --block-size only selects which
# specialisation is compiled.

import triton
import triton.language as tl


@triton.jit
def init_kernel(a_ptr, b_ptr, c_ptr, init_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    initA = tl.load(init_ptr + 0)
    initB = tl.load(init_ptr + 1)
    initC = tl.load(init_ptr + 2)
    tl.store(a_ptr + offs, tl.full((BLOCK_SIZE,), 0, initA.dtype) + initA)
    tl.store(b_ptr + offs, tl.full((BLOCK_SIZE,), 0, initB.dtype) + initB)
    tl.store(c_ptr + offs, tl.full((BLOCK_SIZE,), 0, initC.dtype) + initC)


@triton.jit
def copy_kernel(a_ptr, c_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(c_ptr + offs, tl.load(a_ptr + offs))


@triton.jit
def mul_kernel(b_ptr, c_ptr, scalar_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    scalar = tl.load(scalar_ptr)
    c = tl.load(c_ptr + offs)
    tl.store(b_ptr + offs, scalar * c)


@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    tl.store(c_ptr + offs, a + b)


@triton.jit
def triad_kernel(a_ptr, b_ptr, c_ptr, scalar_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    scalar = tl.load(scalar_ptr)
    b = tl.load(b_ptr + offs)
    c = tl.load(c_ptr + offs)
    tl.store(a_ptr + offs, b + scalar * c)


@triton.jit
def dot_kernel(a_ptr, b_ptr, partial_ptr, array_size, num_programs,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.load(a_ptr).dtype)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    stride = BLOCK_SIZE * num_programs
    while tl.min(offs) < array_size:
        mask = offs < array_size
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        acc = acc + a * b
        offs += stride

    tl.store(partial_ptr + pid, tl.sum(acc, axis=0))
