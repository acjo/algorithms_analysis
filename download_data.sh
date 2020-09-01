#!/bin/bash
# Download data files and place them in the lab folders.

SOURCE="https://github.com/Foundations-of-Applied-Mathematics/Data.git"
PYTHONESSENTIALS=("Exceptions_FileIO" "Profiling")
DATASCIENCEESSENTIALS=("UnixShell1" "SQL1" "SQL2")
VOLUME1=("Conditioning_Stability" "Differentiation" "DrazinInverse" "FacialRecognition" "ImageSegmentation" "LeastSquares_Eigenvalues" "LinearTransformations" "PageRank" "SVD_ImageCompression")
GIT="https://git-scm.com"
TEMPDIR="_DATA_"`date +%s`"_"

# Check that git is installed.
command -v git > /dev/null ||
{ echo -e "\nERROR: git is required. Download it at $GIT.\n"; exit 1; }

# Download the data using git sparse checkout.
mkdir ${TEMPDIR}
cd ${TEMPDIR}
git init --quiet
git remote add origin "${SOURCE}"
git config core.sparseCheckout true
for lab in ${PYTHONESSENTIALS[@]}; do
    echo "${lab}" >> .git/info/sparse-checkout
done
for lab in ${DATASCIENCEESSENTIALS[@]}; do
    echo "${lab}" >> .git/info/sparse-checkout
done
for lab in ${VOLUME1[@]}; do
    echo "${lab}" >> .git/info/sparse-checkout
done
echo -e "\nInitializing Download ...\n"
git pull origin master
cd ../

# Migrate the files from the temporary folder.
set +e
echo -e "\nMigrating files ..."
for lab in ${PYTHONESSENTIALS[@]}; do
    # Check that the target directory exists before copying.
    if [ -d "./${lab}" ]; then
        cp -v ${TEMPDIR}/PythonEssentials/${lab}/* ./${lab}/
    else
        echo -e "\nERROR: directory '${lab}' not found!\n"
    fi
done
for lab in ${DATASCIENCEESSENTIALS[@]}; do
    # Check that the target directory exists before copying.
    if [ -d "./${lab}" ]; then
        cp -rv ${TEMPDIR}/DataScienceEssentials/${lab}/* ./${lab}/
    else
        echo -e "\nERROR: directory '${lab}' not found!\n"
    fi
done
for lab in ${VOLUME1[@]}; do
    # Check that the target directory exists before copying.
    if [ -d "./${lab}" ]; then
        cp -v ${TEMPDIR}/Volume1/${lab}/* ./${lab}/
    else
        echo -e "\nERROR: directory '${lab}' not found!\n"
    fi
done

# Delete the temporary folder.
rm -rf ${TEMPDIR}
echo -e "\nDone.\n"
