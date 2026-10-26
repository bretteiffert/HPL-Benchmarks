#!/usr/bin/env bash

set -euo pipefail

show_help() {
    cat <<'EOF'
Usage:
  ./run.sh --stencil
  ./run.sh --babelstream
  ./run.sh --hartree-fock
  ./run.sh --miniBUDE
  ./run.sh --all
  ./run.sh --help
EOF
}

run_stencil() { 
    bash hip/seven-point-stencil/run_hip.sh >> output/stencil/hip_block_sweep.txt;
    #bash cubecl/seven-point-stencil/run_hip.sh >> output/stencil/cubecl_hip_block_sweep.txt;
    #bash kernelabstractions/seven-point-stencil/run_hip.sh >> output/stencil/ka_hip_block_sweep.txt
    #ash Mojo/seven-point-stencil/run_hip.sh >> output/stencil/Mojo_hip_block_sweep.txt;
    #bash triton/seven-point-stencil/run_hip.sh >> output/stencil/triton_hip_block_sweep.txt;

}

run_babelstream() {
    bash hip/babelstream/run_hip.sh >> output/babelstream/hip_hip.txt;
    #bash cubecl/babelstream/run_hip.sh >> output/babelstream/cubecl_hip.txt;
    #bash kernelabstractions/babelstream/run_hip.sh >> output/babelstream/ka_hip.txt
    #bash Mojo/babelstream/run_hip.sh >> output/babelstream/Mojo_hip.txt;
    #bash triton/babelstream/run_hip.sh >> output/babelstream/triton_hip.txt;
}

run_hartreefock() {
    bash hip/hartree-fock/run_hip.sh >> output/hartree-fock/hip_hip.txt;
    bash cubecl/hartree-fock/run_hip.sh >> output/hartree-fock/cubecl_hip.txt;
    #bash kernelabstractions/hartree-fock/run_hip.sh >> output/hartree-fock/ka_hip.txt;
    #bash Mojo/hartree-fock/run_hip.sh >> output/hartree-fock/Mojo_hip.txt;
    #bash triton/hartree-fock/run_hip.sh >> output/hartree-fock/triton_hip.txt;
}

run_minibude() {
    bash hip/miniBUDE/run_hip.sh >> output/miniBUDE/hip_hip.txt;
    #bash cubecl/miniBUDE/run_hip.sh >> output/miniBUDE/cubecl_hip.txt;
    #bash kernelabstractions/miniBUDE/run_hip.sh >> output/miniBUDE/ka_hip.txt;
    #bash Mojo/miniBUDE/run_hip.sh >> output/miniBUDE/Mojo_hip.txt;
    #bash triton/miniBUDE/run_hip.sh >> output/miniBUDE/triton_hip.txt;
}

run_all() {
    run_stencil;
    run_babelstream;
    run_hartreefock;
    run_minibude;
}

if [[ $# -eq 0 ]]; then
    show_help
    exit 1
fi

case "$1" in
    --stencil)
        run_stencil
        ;;
    --babelstream)
        run_babelstream
        ;;
    --hartree-fock)
        run_hartreefock
        ;;
    --miniBUDE)
        run_minibude
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
