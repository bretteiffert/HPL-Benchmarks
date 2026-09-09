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
    source hudson_setup.sh   
    bash cuda/seven-point-stencil/run_cuda.sh >> output/stencil/cuda.txt;
    bash cubecl/seven-point-stencil/run_cuda.sh >> output/stencil/cubecl_cuda.txt;
    bash julia/ka/seven-point-stencil/cuda/run_cuda.sh >> output/stencil/ka_cuda.txt;
    bash Mojo/seven-point-stencil/run_cuda.sh >> output/stencil/Mojo_cuda.txt;
    bash triton/seven-point-stencil/run_cuda.sh >> output/stencil/triton_cuda.txt;

}

run_babel_stream() {
    source hudson_setup.sh
    bash cuda/babelStream/run_cuda.sh >> output/babelstream/cuda_cuda.txt;
    bash cubecl/babelStream/run_cuda.sh >> output/babelstream/cubecl_cuda.txt;
    bash julia/ka/babelStream/cuda/run_cuda.sh >> output/babelstream/ka_cuda.txt;
    bash Mojo/babelStream/run_cuda.sh >> output/babelstream/Mojo_cuda.txt;
    bash triton/babelStream/run_cuda.sh >> output/babelstream/triton_cuda.txt;
}

run_hartree_fock() {
    source hudson_setup.sh
    bash cuda/hartree-fock/run_cuda.sh >> output/hartree-fock/cuda_cuda.txt;
    bash cubecl/hartree-fock/run_cuda.sh >> output/hartree-fock/cubecl_cuda.txt;
    bash julia/ka/hartree-fock/cuda/run_cuda.sh >> output/hartree-fock/ka_cuda.txt;
    bash Mojo/hartree-fock/run_cuda.sh >> output/hartree-fock/Mojo_cuda.txt;
    bash triton/hartree-fock/run_cuda.sh >> output/hartree-fock/triton_cuda.txt;
}

run_mini_bude() {
    source hudson_setup.sh
    bash cuda/miniBUDE/run_cuda.sh >> output/miniBUDE/cuda_cuda.txt;
    bash cubecl/miniBUDE/run_cuda.sh >> output/miniBUDE/cubecl_cuda.txt;
    bash julia/ka/miniBUDE/cuda/run_cuda.sh >> output/miniBUDE/ka_cuda.txt;
    bash Mojo/miniBUDE/run_cuda.sh >> output/miniBUDE/Mojo_cuda.txt;
    bash triton/miniBUDE/run_cuda.sh >> output/miniBUDE/triton_cuda.txt;
}

run_all() {
    run_stencil;
    run_babel_stream;
    run_hartree_fock;
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
