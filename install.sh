#!/bin/bash

# Install:
# chmod +x install.sh
# conda init bash
# ./install.sh

script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
root_dir=$current_dir
# root_dir=$(dirname "$current_dir")

# Add channels
echo -e "\nAdding channels"
conda config --append channels conda-forge
conda config --append channels nvidia
conda config --append channels pytorch
echo -e "... Done"

# Install `mamba`
echo -e "\nInstalling 'mamba':"
conda install -c conda-forge mamba --y
echo -e "... Done"

# Update 'base' env
echo -e "\nUpdating 'base' environment:"
conda update --a --y
pip install --upgrade pip
echo -e "... Done"

# Install 'mon' env
case "$OSTYPE" in
  linux*)
    echo -e "\nLinux / WSL"
    # Create `mon` env
    env_yml_path="${current_dir}/linux.yml"
    if { conda env list | grep 'mon'; } >/dev/null 2>&1; then
      echo -e "\nUpdating 'mon' environment:"
      conda env update --name mon -f "${env_yml_path}"
      echo -e "... Done"
    else
      echo -e "\nCreating 'mon' environment:"
      conda env create -f "${env_yml_path}"
      echo -e "... Done"
    fi
    eval -e "$(conda shell.bash hook)"
    conda activate mon
    pip install --upgrade pip
    # Remove `cv2/plugin` folder
    rm -rf $CONDA_PREFIX/lib/python3.10/site-packages/cv2/qt/plugins
    ;;
  darwin*)
    echo -e "\nMacOS"
    # Must be called before installing PyTorch Lightning
    export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
    export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1
    # Create `mon` env
    env_yml_path="${current_dir}/mac.yml"
    if { conda env list | grep 'mon'; } >/dev/null 2>&1; then
      echo -e "\nUpdating 'mon' environment:"
      conda env update --name mon -f "${env_yml_path}"
      echo -e "... Done"
    else
      echo -e "\nCreating 'mon' environment:"
      conda env create -f "${env_yml_path}"
      echo -e "... Done"
    fi
    eval "$(conda shell.bash hook)"
    conda activate mon
    pip install --upgrade pip
    # Remove `cv2/plugin` folder:
    rm -rf $CONDA_PREFIX/lib/python3.10/site-packages/cv2/qt/plugins
    ;;
  win*)
    echo -e "\nWindows"
    # Create `mon` env
    env_yml_path="${current_dir}/linux.yml"
    if { conda env list | grep 'mon'; } >/dev/null 2>&1; then
      echo -e "\nUpdating 'mon' environment:"
      conda env update --name mon -f "${env_yml_path}"
      echo -e "... Done"
    else
      echo -e "\nCreating 'mon' environment:"
      conda env create -f "${env_yml_path}"
      echo -e "... Done"
    fi
    eval "$(conda shell.bash hook)"
    conda activate mon
    pip install --upgrade pip
    # Remove `cv2/plugin` folder:
    rm -rf $CONDA_PREFIX/lib/python3.10/site-packages/cv2/qt/plugins
    ;;
  msys*)
    echo -e "\nMSYS / MinGW / Git Bash"
    ;;
  cygwin*)
    echo -e "\nCygwin"
    ;;
  bsd*)
    echo -e "\nBSD"
     ;;
  solaris*)
    echo -e "\nSolaris"
    ;;
  *)
    echo -e "\nunknown: $OSTYPE"
    ;;
esac

# Install 'mon' package
conda activate mon
poetry install

# Set environment variables
# shellcheck disable=SC2162
echo -e "\nSetting DATA_DIR"
data_dir="/data"
if [ ! -d "$data_dir" ];
then
  data_dir="${root_dir}/data"
fi
read -e -i "$data_dir" -p "Enter DATA_DIR=" input
data_dir="${input:-$data_dir}"
if [ "$data_dir" != "" ]; then
  export DATA_DIR="$data_dir"
  mamba env config vars set data_dir="$data_dir"
  echo -e "\nDATA_DIR has been set to $data_dir."
else
  echo -e "\nDATA_DIR has NOT been set."
fi
if [ -d "$root_dir" ];
then
  echo -e "\nDATA_DIR=$data_dir" > "${root_dir}/pycharm.env"
fi
echo -e "... Done"

# Setup Resilio Sync
echo -e "\nSetting Resilio Sync"
rsync_dir="${root_dir}/.sync"
mkdir -p "${rsync_dir}"
cp "IgnoreList" "${rsync_dir}/IgnoreList"
echo -e "... Done"

# Cleanup everything
conda clean --a --y
conda activate mon
