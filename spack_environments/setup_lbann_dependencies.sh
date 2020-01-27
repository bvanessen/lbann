#!/bin/sh

# This script is expected to be called from build_lbann.sh or to have the following environment variables defined
# LBANN_HOME
# LBANN_BUILD_DIR
# SPACK_ENV_DIR
# CENTER

CMD="spack env create -d ${LBANN_BUILD_DIR} ${SPACK_ENV_DIR}/${SPACK_ENV}"
echo ${CMD}
${CMD}
CMD="cp ${SPACK_ENV_DIR}/std_versions_and_variants.yaml ${LBANN_BUILD_DIR}/"
echo ${CMD}
${CMD}
CMD="cp ${SPACK_ENV_DIR}/${CENTER}/externals-${SPACK_ARCH}.yaml ${LBANN_BUILD_DIR}/externals.yaml"
echo ${CMD}
${CMD}
CMD="cp ${SPACK_ENV_DIR}/${CENTER}/compilers.yaml ${LBANN_BUILD_DIR}/"
echo ${CMD}
${CMD}
cd ${LBANN_BUILD_DIR}
echo "********************************************************************************"
echo "* WARNING a fresh installation of the spack dependencies can take a long time   "
echo "********************************************************************************"
CMD="spack install"
echo ${CMD}
${CMD}
RESULT=$?
if [ $RESULT -ne 0 ]; then
    echo "Spack installation failed"
    exit -1
fi
CMD="spack env loads" # Spack creates a file named loads that has all of the correct modules
echo ${CMD}
${CMD}
source ${SPACK_ROOT}/share/spack/setup-env.sh # Rerun setup since spack doesn't modify MODULEPATH unless there are module files defined
CMD="source ${PWD}/loads"
echo ${CMD}
${CMD}