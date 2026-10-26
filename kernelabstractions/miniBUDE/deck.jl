# deck.jl — miniBUDE input deck: ligand.in, protein.in, forcefield.in, poses.in
# (binary, as written by upstream miniBUDE) and ref_energies.out (text).
# All problems are reported as DeckError, never a crash.

module Deck

export DeckError, DiskAtom, DiskFFParams, DeckData, read_deck, nposes, natlig, natpro, ntypes

struct DeckError <: Exception
    msg::String
end
Base.showerror(io::IO, e::DeckError) = print(io, e.msg)

struct DiskAtom
    x::Float32
    y::Float32
    z::Float32
    type::Int32
end

struct DiskFFParams
    hbtype::Int32
    radius::Float32
    hphb::Float32
    elsc::Float32
end

struct DeckData
    dir::String
    protein::Vector{DiskAtom}
    ligand::Vector{DiskAtom}
    forcefield::Vector{DiskFFParams}
    poses::Vector{Vector{Float32}}      # 6 arrays of nposes each
    ref_energies::Vector{Float32}
    max_poses::Int
end

nposes(d::DeckData) = length(d.poses[1])
natlig(d::DeckData) = length(d.ligand)
natpro(d::DeckData) = length(d.protein)
ntypes(d::DeckData) = length(d.forcefield)

function _read_structs(::Type{T}, path::String) where {T}
    isfile(path) || throw(DeckError("deck file not found: $path"))
    bytes = try
        read(path)
    catch e
        throw(DeckError("cannot read $path: $(sprint(showerror, e))"))
    end
    length(bytes) % sizeof(T) == 0 ||
        throw(DeckError("$path is $(length(bytes)) bytes, not a multiple of the " *
                        "$(sizeof(T))-byte record size"))
    isempty(bytes) && throw(DeckError("$path is empty"))
    return collect(reinterpret(T, bytes))
end

"Read the deck in `dir`; `poses == 0` means all poses in the deck."
function read_deck(dir::String, poses::Int)
    isdir(dir) || throw(DeckError("deck directory not found: $dir"))
    ligand = _read_structs(DiskAtom, joinpath(dir, "ligand.in"))
    protein = _read_structs(DiskAtom, joinpath(dir, "protein.in"))
    forcefield = _read_structs(DiskFFParams, joinpath(dir, "forcefield.in"))

    nt = length(forcefield)
    for (name, atoms) in (("ligand.in", ligand), ("protein.in", protein))
        for (k, a) in enumerate(atoms)
            0 <= a.type < nt ||
                throw(DeckError("$(joinpath(dir, name)): atom $k has type $(a.type); " *
                                "forcefield.in has types 0..$(nt - 1)"))
        end
    end

    flat = _read_structs(Float32, joinpath(dir, "poses.in"))
    length(flat) % 6 == 0 ||
        throw(DeckError("$(joinpath(dir, "poses.in")) holds $(length(flat)) values, " *
                        "not a multiple of 6"))
    max_poses = length(flat) ÷ 6
    n = poses == 0 ? max_poses : poses
    n <= max_poses ||
        throw(DeckError("requested $n poses but the deck has $max_poses"))
    pose_arrays = [flat[((k-1)*max_poses+1):((k-1)*max_poses+n)] for k in 1:6]

    ref_path = joinpath(dir, "ref_energies.out")
    isfile(ref_path) || throw(DeckError("deck file not found: $ref_path"))
    ref = Float32[]
    for (ln, line) in enumerate(eachline(ref_path))
        s = strip(line)
        isempty(s) && continue
        v = tryparse(Float32, s)
        v === nothing && throw(DeckError("$ref_path line $ln: '$s' is not a number"))
        push!(ref, v)
    end
    n <= length(ref) ||
        throw(DeckError("$ref_path has $(length(ref)) energies, fewer than the $n poses requested"))

    return DeckData(dir, protein, ligand, forcefield, pose_arrays, ref, max_poses)
end

end # module Deck
