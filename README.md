# NVIDIA eGPU on Linux

Notes on how to use NVIDIA eGPUs on Linux.

Table of contents:

- [NVIDIA eGPU on Linux](#nvidia-egpu-on-linux)
  - [Install Linux](#install-linux)
    - [Step 0: Hardware Requirements](#step-0-hardware-requirements)
    - [Step 1: Install Ubuntu](#step-1-install-ubuntu)
    - [Step 2: Install Basic Development Software via CLI](#step-2-install-basic-development-software-via-cli)
    - [Step 3: Install and Configure NVIDIA and GPU Related Libraries](#step-3-install-and-configure-nvidia-and-gpu-related-libraries)
      - [eGPU Switcher](#egpu-switcher)
    - [Step 4: Check](#step-4-check)
    - [Step 5: (Optional) Basic Additional Configuration](#step-5-optional-basic-additional-configuration)
  - [Using the GPU](#using-the-gpu)
    - [Python Environment](#python-environment)
    - [Pytorch Example](#pytorch-example)
  - [Further Useful Applications](#further-useful-applications)
    - [VSCode Server](#vscode-server)
    - [Ollama Server](#ollama-server)
  - [Extra: Minimum Personal Migration Checklist](#extra-minimum-personal-migration-checklist)
    - [Minimum Software Setup](#minimum-software-setup)
    - [Minimum VSCode Extensions](#minimum-vscode-extensions)
    - [Configuration](#configuration)
    - [External SSD for Storage](#external-ssd-for-storage)
  - [Sources, Related Links](#sources-related-links)

## Install Linux

### Step 0: Hardware Requirements

- [Lenovo ThinkPad P14s Gen 2i](https://www.lenovo.com/gb/en/p/laptops/thinkpad/thinkpadp/p14s-amd-g1/22wsp144sa1)
	- **Graphics Card (GPU): Quadro T500 Mobile**
	- 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz, x86_64
	- NVMe SSD: SK Hynix 512 GB
- [Gigabyte NVIDIA GeForce RTX 3060 Gaming OC 12 GB V2 LHR](https://www.gigabyte.com/Graphics-Card/GV-N3060GAMING-OC-12GD-rev-20)
- [Razer Core X External Case for Thunderbolt 3 Graphics Card](https://www.razer.com/mena-en/gaming-laptops/razer-core-x)



- Thunderbolt 3 or superior: here's where we're going to connect our external GPU.
- NVIDIA GPU chip: devices which have graphics cards from other vendors than NVIDIA tend to have issues when NVIDIA drivers are installed; therefore, I would recommend a laptop which already has an integrated NVIDIA GPU, even though we will use the external, more powerful one.

### Step 1: Install Ubuntu




Summary of the steps to follow:

- [Donwload the Ubuntu Desktop](https://ubuntu.com/download/desktop) version for you architecture. In my case, I downloaded version 25.04 for Intel/AMD64.
- Create a Flash drive with the downloaded image using [Balena Etcher](https://etcher.balena.io/).
- [Install Ubuntu using the flashed USB drive](https://canonical-ubuntu-desktop-documentation.readthedocs-hosted.com/en/latest/tutorial/install-ubuntu-desktop/):
  - Back up any files or data you have on your computer.
  - Power off; then, power on while pressing F12: we will access the UEFI BIOS. Usually F12 is required to access the BIOS, but you might need to try other keys if F12 doesn't work.
  - Select USB.
  - Select "Try or Install Ubuntu".
  - Follow instructions.
  - Allow 3rd party proprietary drivers (e.g., NVIDIA).
  - Erase and install Ubuntu (no special formats like LVM, ZFS, no encryption): note that this will
  - No Active Directory
  - Bitlocker warnings: I decided to wipe out the disk.

If we need to upgrade to a newer version (e.g., [from 24.04 to 25.04](https://www.omgubuntu.co.uk/2024/10/how-to-upgrade-to-ubuntu-24-10)), we 



```bash
# Configure *Software Updater* (in App Center): 
# Settings > Updates > Notify me: Any new version
sudo apt update
sudo apt full-upgrade

# Check release 25.04
lsb_release -a
```

If we face any issues regarding the access to eGPU via Thunderbolt.


1. In the UEFI BIOS menu, find a *Security* tab/option, and set the Thunderbolt security option to be *Unrestricted*.
2. Ubuntu Settings UI: Privacy & Security > Thunderbolt: *allow access to devices*, such as GPUs.




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

# -- Docker

# Prerequisites
sudo apt update
sudo apt install ca-certificates curl gnupg lsb-release -y

# Add official Docker's GPG Key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add Docker repo
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine + CLI + Compose
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y

# Enable and start docker service
sudo systemctl enable docker
sudo systemctl start docker

# Check docker is running
sudo systemctl status docker

# Add your our user to the docker group to run without sudo
sudo groupadd -f docker
sudo usermod -aG docker $USER
newgrp docker
groups  # verify docker appears as our group

# Test
docker run hello-world

# Enable NVIDIA support
sudo apt update


sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test GPU passthrough
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

```

### Step 3: Install and Configure NVIDIA and GPU Related Libraries

Apps: Open Software & Updates
- Settings
- Additional drivers: NVIDIA selected
	- Using NVIDIA driver nvidia-driver-580-open
	- Driver version 580 is the latest, but that lead to random freezes on Ubuntu 24.04
- If no NVIDIA available, follow https://gist.github.com/tanmayyb/d19f9aa5641349f8830d05e2c91d5a79
	- Disable Nouveau drivers
- Install NVIDIA stuff via CLI

```bash
# Use this, otherwise freezing problems
sudo apt install nvidia-driver-580-open

# Tools
sudo apt-get nvidia-smi
sudo apt-get install nvidia-settings
```

Install the CUDA Toolkit. In addition to the drivers, we sometimes need the toolkit:
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local
Linux / x86_64 / Ubuntu / 24.04 / deb (local)


```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin

sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/13.0.1/local_installers/cuda-repo-ubuntu2404-13-0-local_13.0.1-580.82.07-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu2404-13-0-local_13.0.1-580.82.07-1_amd64.deb

sudo cp /var/cuda-repo-ubuntu2404-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/

sudo apt-get update

sudo apt-get -y install cuda-toolkit-13-0

sudo apt install nvidia-cuda-toolkit
```

#### eGPU Switcher

- https://github.com/hertg/egpu-switcher
- https://github.com/hertg/egpu-switcher/releases (0.20.1 in my case)

```bash
sudo cp Downloads/egpu-switcher-amd64 /opt/egpu-switcher
sudo chmod 755 /opt/egpu-switcher
sudo ln -s /opt/egpu-switcher /usr/bin/egpu-switcher
sudo egpu-switcher enable
```

### Step 4: Check


```bash
# Single call
nvidia-smi

# Loop call for refeshed outputs
nvidia-smi -l 1
```

### Step 5: (Optional) Basic Additional Configuration

Settings: System: activate remote access options:

- Secure Shell
- Remote Desktop


## Using the GPU

```bash
# Single call
nvidia-smi

# Loop call for refeshed outputs
nvidia-smi -l 1
```

### Python Environment

One of the easiest ways to check 

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

```bash
pip uninstall bitsandbytes
pip install bitsandbytes
```

### Pytorch Example


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

## Further Useful Applications




### VSCode Server

### Ollama Server

```bash
# Install Ollama CLI and a the service ollama.service
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model to ~/.ollama/models
# The following models fit well with the RTX 3060
ollama pull llama3:8b
ollama pull mistral
ollama pull gemma:7b
# Run the service
# By default, tt should be running in OLLAMA_HOST=127.0.0.1:11434
ollama run llama3:8b "Hello, are you running on my GPU?"

# We can chat in the CLI or use cURL
curl http://127.0.0.1:11434/api/generate -d '{
  "model": "llama3:8b",
  "prompt": "Write me a haiku about Ubuntu and GPUs."
}'

# Verify GPU usage
ollama ps

# To stop AND OFFLOAD weights from GPU
ollama stop llama3:8b

# Force GPU usage
export OLLAMA_USE_GPU=1
ollama run llama3:8b

# If we want, we can start a/the service manually.
# If we have problems with the default port 11434,
# we can change it to 11435
export OLLAMA_HOST=127.0.0.1:11435
ollama serve &
# To stop
ps -ef | grep ollama # get PID
kill -9 <PID>

# Check the service is running
systemctl --user status ollama

# Stop ollama service
systemctl --user stop ollama

# Restart service
systemctl --user restart ollama

# Enable/disable autostart when login
systemctl --user enable ollama
systemctl --user disable ollama
```


## Extra: Minimum Personal Migration Checklist

### Minimum Software Setup

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
- Browser Bookmarks (+ log in with Bitwarden)
  - Evernote
  - ChatGPT
  - gMail
  - ProtonMail
  - GitHub
  - SimpleLogin

### Minimum VSCode Extensions

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

Change Terminal appearance: `Hamburger menu > Preferences`:

- Select Profile: Unnamed / Mikel
- Tab: Colors
  - De-select *"Use colors from system theme"*
  - Built-in schemes: select one, e.g.: *GNOME dark*

Set at least a basic `~/.gitconfig`:

```
[user]
    name = Your Name
    email = your.email@example.com
```

### External SSD for Storage

I have a [Crucial X9 1TB Portable SSD](https://www.crucial.com/ssd/x9/CT1000X9SSD9).

Add a new USB SSD and format it to be `poseidon`.

`Apps: Disks`: Select partition and format as Ext4.

Then, create a data folder in it and link the home data folder to it:

```bash
# Check SSD is there: poseidon
lsblk -o NAME,MOUNTPOINT,LABEL

# Create a symbolic link
mkdir -p /media/$USER/poseidon/data
ln -s /media/$USER/poseidon/data ~/data
```

## Sources, Related Links

- https://gist.github.com/tanmayyb/d19f9aa5641349f8830d05e2c91d5a79
- https://www.reddit.com/r/framework/comments/11dtm78/guide_for_setting_up_egpu_with_framework_11th_gen/
- https://gist.github.com/valteu/1c0a9b7288cc3d77a6654a4d22d0ce9f
