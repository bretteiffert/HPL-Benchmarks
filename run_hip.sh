#!/usr/bin/env bash

set -euo pipefail

show_help() {
    cat <<'EOF'
Usage:
  ./run.sh --stencil
  ./run.sh --babelStream
  ./run.sh --hartree-fock
  ./run.sh --miniBUDE
  ./run.sh --all
  ./run.sh --help
EOF
}

run_stencil() { 
    bash hip/seven-point-stencil/run_hip.sh >> output/stencil/hip_block_sweep.txt;
    bash cubecl/seven-point-stencil/run_hip.sh >> output/stencil/cubecl_hip_block_sweep.txt;
    bash julia/ka/seven-point-stencil/hip/run_hip.sh >> output/stencil/ka_hip_block_sweep.txt
    bash Mojo/seven-point-stencil/run_hip.sh >> output/stencil/Mojo_hip_block_sweep.txt;
    bash triton/seven-point-stencil/run_hip.sh >> output/stencil/triton_hip_block_sweep.txt;

}

run_babel_stream() {
    bash hip/babelStream/run_hip.sh >> output/babelstream/hip_hip.txt;
    bash cubecl/babelStream/run_hip.sh >> output/babelstream/cubecl_hip.txt;
    bash julia/ka/babelStream/hip/run_hip.sh >> output/babelstream/ka_hip.txt
    bash Mojo/babelStream/run_hip.sh >> output/babelstream/Mojo_hip.txt;
    bash triton/babelStream/run_hip.sh >> output/babelstream/triton_hip.txt;
}

run_hartree_fock() {
    bash hip/hartree-fock/run_hip.sh >> output/hartree-fock/hip_hip.txt;
    bash cubecl/hartree-fock/run_hip.sh >> output/hartree-fock/cubecl_hip.txt;
    bash julia/ka/hartree-fock/hip/run_hip.sh >> output/hartree-fock/ka_hip.txt;
    bash Mojo/hartree-fock/run_hip.sh >> output/hartree-fock/Mojo_hip.txt;
    bash triton/hartree-fock/run_hip.sh >> output/hartree-fock/triton_hip.txt;
}

run_mini_bude() {
    bash hip/miniBUDE/run_hip.sh >> output/miniBUDE/hip_hip.txt;
    bash cubecl/miniBUDE/run_hip.sh >> output/miniBUDE/cubecl_hip.txt;
    bash julia/ka/miniBUDE/hip/run_hip.sh >> output/miniBUDE/ka_hip.txt;
    bash Mojo/miniBUDE/run_hip.sh >> output/miniBUDE/Mojo_hip.txt;
    bash triton/miniBUDE/run_hip.sh >> output/miniBUDE/triton_hip.txt;
}

run_all() {
    run_stencil;
    run_babel_stream;
    run_hartree_fock;
    run_mini_bude;
}

if [[ $# -eq 0 ]]; then
    show_help
    exit 1
fi

case "$1" in
    --stencil)
        run_stencil
        ;;
    --babelStream)
        run_babel_stream
        ;;
    --hartree-fock)
        run_hartree_fock
        ;;
    --miniBUDE)
        run_mini_bude
        ;;
    --all)
        run_all
        ;;
    --help|-h)
        show_help
        ;;
    *)
        echo "Error: Unknown option: $1" >&2
        show_help >&2
        exit 1
        ;;
esac
