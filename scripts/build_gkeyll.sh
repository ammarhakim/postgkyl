#!/bin/sh
set -eu

usage() {
    cat <<EOF
usage: sh $0 [-h]

Build the selected Gkeyll module and its prerequisites in gkeyll/, obtaining the
checkout only when absent. Local edits are built as-is; updating upstream is
a separate operation (scripts/update_gkeyll.sh).

options:
  -h, --help   show this help message and exit

configuration:
  postgkyl-build.toml selects the module and build ownership.

environment:
  POSTGKYL_BUILD_CONFIG  alternate configuration file
  PYTHON       interpreter for reading configuration (default: python)
  POSTGKYL_GKEYLL_MODULE  highest module: core (default), moments, vlasov, gyrokinetic, pkpm
  CC           C compiler (default: cc)
  ARCH_FLAGS   architecture flags (default: none, for portable binaries)
  BUILD_JOBS   parallel make jobs (default: half the processors)
EOF
}
case "$#:${1:-}" in
    0:) ;;
    1:-h|1:--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
GKEYLL_DIR="${ROOT_DIR}/gkeyll"
GKEYLL_MODULE=$("${PYTHON:-python}" "${SCRIPT_DIR}/gkeyll_config.py" --module) || exit 2
MODULES=""
while IFS= read -r module; do
    MODULES="${MODULES} ${module}"
    if [ "$module" = "$GKEYLL_MODULE" ]; then break; fi
done < "${SCRIPT_DIR}/gkeyll-modules"
if [ ! -e "${GKEYLL_DIR}/.git" ]; then
    sh "${SCRIPT_DIR}/update_gkeyll.sh"
fi

# Expand sparse checkouts without removing other sources or changing revisions.
if [ "$(git -C "${GKEYLL_DIR}" config --bool core.sparseCheckout || true)" = true ]; then
    for module in $MODULES; do
        git -C "${GKEYLL_DIR}" sparse-checkout add "/${module}/"
    done
fi

CC="${CC:-cc}"
LAPACK_LITE=yes
BUILD_DIR=build
if [ "$GKEYLL_MODULE" != core ]; then
    LAPACK_LITE=no
    BUILD_DIR=build/postgkyl-full
fi
echo "# Configuring gkeyll (CC=${CC}, lapack-lite=${LAPACK_LITE}, app=${GKEYLL_MODULE})"
(cd "${GKEYLL_DIR}" && ./configure "CC=${CC}" --use-lapack-lite="${LAPACK_LITE}" --app="${GKEYLL_MODULE}")

ARCH_FLAGS="${ARCH_FLAGS:-}"
export ARCH_FLAGS

echo "# Building ${GKEYLL_MODULE} and prerequisites (ARCH_FLAGS=${ARCH_FLAGS:-<none -- compiler default>})"
if [ -z "${BUILD_JOBS:-}" ]; then
    BUILD_JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)
    BUILD_JOBS=$((BUILD_JOBS > 1 ? BUILD_JOBS / 2 : 1))
fi
(cd "${GKEYLL_DIR}" && make "${GKEYLL_MODULE}" LIBRARY_TARGET=library "BUILD_DIR=${BUILD_DIR}" "ARCH_FLAGS=${ARCH_FLAGS}" \
    -j"${BUILD_JOBS}")

for module in $MODULES; do
    SO_PATH="${GKEYLL_DIR}/${BUILD_DIR}/${module}/libg0${module}.so"
    if [ ! -f "${SO_PATH}" ]; then
        echo "error: expected ${SO_PATH} after build, but it is missing" >&2
        exit 1
    fi
    echo "# Built ${SO_PATH}"
done
