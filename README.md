# NVIDIA eGPU on Linux

Notes on how to use NVIDIA eGPUs on Linux.

Table of contents:

- [NVIDIA eGPU on Linux](#nvidia-egpu-on-linux)
  - [Install Linux](#install-linux)
    - [Hardware](#hardware)
    - [Step 1: Install Ubuntu](#step-1-install-ubuntu)
    - [Step 2: Install Basic Development Software via CLI](#step-2-install-basic-development-software-via-cli)
    - [Step 3: Install and Configure NVIDIA and GPU Related Libraries](#step-3-install-and-configure-nvidia-and-gpu-related-libraries)
      - [eGPU Switcher](#egpu-switcher)
  - [Check GPU](#check-gpu)
    - [Pytorch](#pytorch)
  - [Useful Applications](#useful-applications)
    - [VSCode Server](#vscode-server)
    - [Ollama Server](#ollama-server)
  - [Extra: Personal Migration Checklist](#extra-personal-migration-checklist)
    - [Software Setup](#software-setup)
    - [VSCode Extensions](#vscode-extensions)
    - [Configuration](#configuration)
    - [External SSD for Storage](#external-ssd-for-storage)

## Install Linux

### Hardware

- [Lenovo ThinkPad P14s Gen 2i](https://www.lenovo.com/gb/en/p/laptops/thinkpad/thinkpadp/p14s-amd-g1/22wsp144sa1)
	- **Graphics Card (GPU): Quadro T500 Mobile**
	- 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz, x86_64
	- NVMe SSD: SK Hynix 512 GB
- [Gigabyte NVIDIA GeForce RTX 3060 Gaming OC 12 GB V2 LHR](https://www.gigabyte.com/Graphics-Card/GV-N3060GAMING-OC-12GD-rev-20)
- [Razer Core X External Case for Thunderbolt 3 Graphics Card](https://www.razer.com/mena-en/gaming-laptops/razer-core-x)

### Step 1: Install Ubuntu

[Install Ubuntu](https://canonical-ubuntu-desktop-documentation.readthedocs-hosted.com/en/latest/tutorial/install-ubuntu-desktop/)



- F12
	- The UEFI BIOS on my laptop is accessed through the F12 key. From Security tab, the thunderbolt security option can be set to: Unrestricted
	- Go back and Select USB
- Try or Install Ubuntu
- Allow 3rd party proprietary drivers (NVIDIA)
- Erase and install Ubuntu (no special formats like LVM, ZFS, no encryption)
- No Active Directory
- Bitlocker: wipe out disk
- [Ubuntu upgrade](https://www.omgubuntu.co.uk/2024/10/how-to-upgrade-to-ubuntu-24-10)


```bash
# Configure *Software Updater* (in App Center): 
# Settings > Updates > Notify me: Any new version
sudo apt update
sudo apt full-upgrade

# Check release 25.04
lsb_release -a
```


### Step 2: Install Basic Development Software via CLI

```bash
sudo apt update
sudo apt-get dist-upgrade

# Basic Tools
sudo apt install -y \
  curl \
  wget \
  git \
  build-essential \
  net-tools \
  unzip \
  htop \
  vim \
  software-properties-common

# Python Tools
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
  python3 \
  python3-dev \
  python3-pip \
  python3-venv \
  python3-setuptools \
  python3-wheel

# C++ Tools
sudo apt install -y build-essential cmake gdb
sudo apt install -y \
  libboost-all-dev \
  libeigen3-dev \
  libopencv-dev

# Optional Dev Extras
sudo apt install -y \
  pkg-config \
  clang \
  valgrind \
  ninja-build
```

### Step 3: Install and Configure NVIDIA and GPU Related Libraries



#### eGPU Switcher



## Check GPU

```bash
# Single call
nvidia-smi

# Loop call for refeshed outputs
nvidia-smi -l 1
```



### Pytorch

First, we need to [install Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install):

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Press Enter through the license.
# Type yes to accept.
# Default install path is ~/miniconda3 (press Enter).
# Say yes if it asks to initialize conda in your shell.

~/miniconda3/bin/conda init bash
source ~/.bashrc
conda init --all
conda --version
cd ~
rm Miniconda3-latest-Linux-x86_64.sh
```

Then, we can create a `gpu` Python environment with the provided [`conda.yaml`](./conda.yaml):

```bash
conda env create -f conda.yaml
conda activate gpu
```




```python
import torch

# How many GPUs?
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

# Name of each GPU
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

# Memory usage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Memory allocated:", torch.cuda.memory_allocated(device) / 1024**2, "MB")
print("Max memory allocated:", torch.cuda.max_memory_allocated(device) / 1024**2, "MB")
print("Memory reserved:", torch.cuda.memory_reserved(device) / 1024**2, "MB")

# Clear cache
torch.cuda.empty_cache()

# Dosplay CLI tool output nvidia-smi
!nvidia-smi

# For live tracking in Terminal, use built-in loop option
# nvidia-smi -l 1
```

## Useful Applications

### VSCode Server

### Ollama Server

## Extra: Personal Migration Checklist

### Software Setup

- [Mullvad VPN](https://mullvad.net/es/download/vpn/linux)
- Ubuntu App Center
  - Brave
  - Bitwarden
  - Sublime
  - VSCode
  - LibreOffice
  - ProtonMail
  - Discord
- Browser Extensions
  - uBlock
  - Bitwarden
  - Evernote
- Browser Bookmarks (+ log in)
  - Evernote
  - ChatGPT
  - gMail
  - ProtonMail
  - GitHub
  - SimpleLogin

### VSCode Extensions

- Python
- C/C++
- Markdown all in one
- Jupyter
- CMake Tools
- DotENV
- Remote-SSH
- Remote Development
- YAML
- LiveShare
- Github Copilot + Copilot Chat
- vscode-icons

### Configuration

Terminal: Hamburger > Preferences

- Select Profile: Unnamed / Mikel
- Tab: Colors
  - Unselect "Use colors from system theme"
  - Built-in schemes: select one, e.g.: GNOME dark

Set `~/.gitconfig`:

```
[user]
    name = Your Name
    email = your.email@example.com
```

### External SSD for Storage

Add a new USB SSD and format it to be poseidon; then create a data folder in it and link the home data folder to it (Apps: Disks -> Select partition and format as Ext4)

```bash
# Check SSD is there: poseidon
lsblk -o NAME,MOUNTPOINT,LABEL

# Create a symbolic link
mkdir -p /media/$USER/poseidon/data
ln -s /media/$USER/poseidon/data ~/data
```
