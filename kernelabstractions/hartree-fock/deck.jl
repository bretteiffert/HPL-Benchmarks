# Hartree-Fock input file ("deck"): number of Gaussians, number of atoms,
# one exponent/coefficient pair per Gaussian, then x y z per atom (Angstrom).
# Values are kept in Float64 exactly as read; the run converts them. Same
# layout and messages as the original read_file / the HIP read_deck.

struct Deck
    path::String
    name::String                 # file name without directories, e.g. "he64"
    ngauss::UInt32
    natoms::UInt32
    xpnt::Vector{Float64}        # ngauss exponents
    coef::Vector{Float64}        # ngauss contraction coefficients
    geom::Vector{Float64}        # 3 * natoms, laid out x[natoms], y[natoms], z[natoms]
end

# A problem with the input file. main prints "error: <what> (try '--help')".
struct DeckError <: Exception
    msg::String
end

function base_name(path::AbstractString)
    p = String(path)
    while ncodeunits(p) > 1 && endswith(p, "/")
        p = p[1:end-1]
    end
    slash = findlast('/', p)
    return slash === nothing ? p : p[nextind(p, slash):end]
end

# Whitespace-separated token reader that follows std::istream operator>>
# (libstdc++): it stops at the first character that cannot continue the number
# and fails on a missing, malformed or out-of-range value.
mutable struct TokenStream
    data::Vector{UInt8}
    pos::Int
    failed::Bool
end

TokenStream(data::Vector{UInt8}) = TokenStream(data, 1, false)

isspace_c(c::UInt8) = c == 0x20 || (0x09 <= c <= 0x0d)
isdigit_c(c::UInt8) = UInt8('0') <= c <= UInt8('9')

function skip_ws!(s::TokenStream)
    while s.pos <= length(s.data) && isspace_c(s.data[s.pos])
        s.pos += 1
    end
end

peek_byte(s::TokenStream) = s.pos <= length(s.data) ? s.data[s.pos] : 0x00

# operator>>(long long&)
function read_ll!(s::TokenStream)::Union{Int64, Nothing}
    s.failed && return nothing
    skip_ws!(s)
    start = s.pos
    c = peek_byte(s)
    (c == UInt8('+') || c == UInt8('-')) && (s.pos += 1)
    d0 = s.pos
    while isdigit_c(peek_byte(s))
        s.pos += 1
    end
    if s.pos == d0
        s.failed = true
        return nothing
    end
    v = tryparse(Int64, String(s.data[start:s.pos-1]))
    if v === nothing                      # out of range: failbit
        s.failed = true
        return nothing
    end
    return v
end

# operator>>(double&): sign, digits, '.', digits, then e/E, sign, digits
# (the exponent only after a mantissa digit); the accumulated text must be a
# complete number and must not overflow.
function read_double!(s::TokenStream)::Union{Float64, Nothing}
    s.failed && return nothing
    skip_ws!(s)
    start = s.pos
    c = peek_byte(s)
    (c == UInt8('+') || c == UInt8('-')) && (s.pos += 1)
    mantissa = false
    while isdigit_c(peek_byte(s))
        s.pos += 1; mantissa = true
    end
    if peek_byte(s) == UInt8('.')
        s.pos += 1
        while isdigit_c(peek_byte(s))
            s.pos += 1; mantissa = true
        end
    end
    complete = mantissa
    if mantissa && (peek_byte(s) == UInt8('e') || peek_byte(s) == UInt8('E'))
        s.pos += 1
        c = peek_byte(s)
        (c == UInt8('+') || c == UInt8('-')) && (s.pos += 1)
        e0 = s.pos
        while isdigit_c(peek_byte(s))
            s.pos += 1
        end
        complete = s.pos > e0
    end
    if !complete
        s.failed = true
        return nothing
    end
    v = tryparse(Float64, String(s.data[start:s.pos-1]))
    if v === nothing || isinf(v)          # overflow: failbit
        s.failed = true
        return nothing
    end
    return v
end

# Reads one positive integer count.
function read_count(s::TokenStream, path::String, what::String)::UInt32
    v = read_ll!(s)
    v === nothing && throw(DeckError("input file '" * path * "': could not read the " * what))
    if v <= 0 || v > 0xFFFFFFFF
        throw(DeckError("input file '" * path * "': the " * what * " must be a positive integer, got " * string(v)))
    end
    return UInt32(v)
end

function read_value(s::TokenStream, path::String, what::String)::Float64
    v = read_double!(s)
    v === nothing && throw(DeckError("input file '" * path * "': missing or malformed " * what))
    isfinite(v) || throw(DeckError("input file '" * path * "': " * what * " is not a finite number"))
    return v
end

# Throws DeckError on a missing, truncated or malformed file.
function read_deck(path::String)::Deck
    data = UInt8[]
    if isdir(path)
        # std::ifstream opens a directory but reads nothing from it
    else
        try
            data = read(path)
        catch
            throw(DeckError("cannot read input file '" * path * "'"))
        end
    end
    s = TokenStream(data)

    ngauss = read_count(s, path, "number of Gaussians")
    natoms = read_count(s, path, "number of atoms")

    xpnt = Vector{Float64}(undef, ngauss)
    coef = Vector{Float64}(undef, ngauss)
    for i in 1:Int(ngauss)
        n = string(i)
        xpnt[i] = read_value(s, path, "exponent of Gaussian " * n)
        coef[i] = read_value(s, path, "coefficient of Gaussian " * n)
        if !(xpnt[i] > 0.0)
            throw(DeckError("input file '" * path * "': exponent of Gaussian " * n * " must be positive"))
        end
    end

    geom = Vector{Float64}(undef, 3 * Int(natoms))
    axis = ("x", "y", "z")
    for i in 0:Int(natoms)-1
        for j in 0:2
            geom[j * Int(natoms) + i + 1] =
                read_value(s, path, axis[j+1] * " coordinate of atom " * string(i + 1))
        end
    end

    return Deck(path, base_name(path), ngauss, natoms, xpnt, coef, geom)
end
