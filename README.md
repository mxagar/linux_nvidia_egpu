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
    - [Step 4: Check the GPU](#step-4-check-the-gpu)
    - [Step 5: Install Docker with NVIDIA GPU Support](#step-5-install-docker-with-nvidia-gpu-support)
    - [Step 6: Remote Access Configuration](#step-6-remote-access-configuration)
  - [Using the GPU](#using-the-gpu)
    - [Setting Up a GPU Python Environment](#setting-up-a-gpu-python-environment)
    - [Quick Pytorch Example](#quick-pytorch-example)
    - [Further Pytorch Tests in the Attached Notebook](#further-pytorch-tests-in-the-attached-notebook)
  - [Further Useful Applications](#further-useful-applications)
    - [VSCode Server: Launch VSCode on Another Machine, but Hosted on the GPU-Ubuntu](#vscode-server-launch-vscode-on-another-machine-but-hosted-on-the-gpu-ubuntu)
    - [Ollama Server: Use Ollama LLMs Running on the GPU-Ubuntu, but from Another Machine](#ollama-server-use-ollama-llms-running-on-the-gpu-ubuntu-but-from-another-machine)
  - [Extra: Minimum Personal Migration Checklist](#extra-minimum-personal-migration-checklist)
    - [Minimum Software Setup](#minimum-software-setup)
    - [Minimum VSCode Extensions](#minimum-vscode-extensions)
    - [Configuration](#configuration)
    - [External SSD for Storage](#external-ssd-for-storage)
  - [Sources, Related Links](#sources-related-links)

## Install Linux

### Step 0: Hardware Requirements

We need:

- a laptop,
- a GPU,
- and an external GPU case to contain the GPU and connect it to the laptop.

Additionally, we should consider these requirements:

- The laptop needs to have Thunderbolt 3 or superior port: here's where we're going to connect our external GPU.
- Also, the laptop needs to have an NVIDIA GPU chip: devices which have graphics cards from other vendors than NVIDIA tend to have issues when NVIDIA drivers are installed; therefore, I would recommend a laptop which already has an integrated NVIDIA GPU, even though we will use the external, more powerful one.
- If you are going to buy an NVIDIA GPU, consider the VRAM necessary to load the models you would like to run, and the amount of data you plan to hold in memory (i.e., the size of a batch); for instance:
  - A batch of 128 RGB images of size 1024 x 1024 can weight between 0.75 GB (FP16) to 1.5 GB (FP32).
  - The object detection model [YOLOv8](https://huggingface.co/Ultralytics/YOLOv8) (with around 68M params) requires between 130 MB (FP16) and 260 MB (FP32) in memory.
  - Size of other vision models:
    - [DETR-R50](https://huggingface.co/facebook/detr-resnet-50) (100 queries): 82 MB (FP16) to 164 (FP32)
    - [DINO-R50](https://huggingface.co/docs/transformers/main/en/model_doc/dinov3): 120 (FP16) to 240 (FP32)
    - [SAM-ViT-H](https://huggingface.co/docs/transformers/en/model_doc/sam): 1.25 GB (FP16) to 2.5 GB (FP32)
  - Size of some popular, local Large Language Models (LLMs):
    - [llama3:8b](https://ollama.com/library/llama3) via [Ollama](https://ollama.com/) requires 4.7 GB
    - [gemma3:12b](https://ollama.com/library/gemma3): 8.1 GB
    - [deepseek-r1:14b](https://ollama.com/library/deepseek-r1) 9.0 GB
    - [gpt-oss:20b](https://ollama.com/library/gpt-oss): 14 GB
    - ...
- Last but not least: Make sure that the external GPU case supports our concrete GPU; cooling, number of pins, powr consumption -- not all combinations are compatile.

Specifically, this is my setup:

- [Lenovo ThinkPad P14s Gen 2i](https://www.lenovo.com/gb/en/p/laptops/thinkpad/thinkpadp/p14s-amd-g1/22wsp144sa1)
	- **Graphics Card (GPU): Quadro T500 Mobile**
	- 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz, x86_64
	- NVMe SSD: SK Hynix 512 GB
- [Gigabyte NVIDIA GeForce RTX 3060 Gaming OC 12 GB V2 LHR](https://www.gigabyte.com/Graphics-Card/GV-N3060GAMING-OC-12GD-rev-20)
- [Razer Core X External Case for Thunderbolt 3 Graphics Card](https://www.razer.com/mena-en/gaming-laptops/razer-core-x)

### Step 1: Install Ubuntu

Summary of the steps to follow:

- [Donwload the Ubuntu Desktop](https://ubuntu.com/download/desktop) version for you architecture. My current version is 25.04 for Intel/AMD64.
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

I originally installed the Ubuntu version `24.04`, but it lead to random freezes. Apparently, that could be related some incompabilities between the Desktop GUI and the NVIDIA drivers.
In case you need to perform an upgrade, you can check [this post](https://www.omgubuntu.co.uk/2024/10/how-to-upgrade-to-ubuntu-24-10).

To upgrade versions via CLI and check the version we have:

```bash
# Configure *Software Updater* (in App Center): 
# Settings > Updates > Notify me: Any new version
sudo apt update
sudo apt full-upgrade

# Check release 25.04
lsb_release -a
```

If we face any issues regarding the access to the eGPU via Thunderbolt, we might need to grant access to it:

1. In the UEFI BIOS menu, look for a *Security* tab/option, and set the Thunderbolt security option to be *Unrestricted*.
2. Ubuntu Settings UI: Privacy & Security > Thunderbolt: *allow access to devices*, such as GPUs.

### Step 2: Install Basic Development Software via CLI

Here, I list the basic development tools I installed via CLI:

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

If we follow the Ubuntu installation as specified and our Laptop has already an integrated NVIDIA chip, we should have the NVIDIA drivers active already; we can check that in the `Software & Updates` app:

- Click on *Settings*
- *Additional drivers*: check that a NVIDIA driver is selected; in my case, I have `nvidia-driver-580-open`.

If that's not the case or we would like to update the drivers manually, we can do that via the CLI:

```bash
# Use this, otherwise freezing problems
sudo apt install nvidia-driver-580-open

# Tools
sudo apt-get nvidia-smi
sudo apt-get install nvidia-settings
```

In addition to the drivers, we also need to [install the CUDA toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local).

The recipe for a setup with `Linux / x86_64 / Ubuntu / 24.04 / deb (local)` is the following:

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

Next, we need to install and enable the [`egpu-switcher`](https://github.com/hertg/egpu-switcher).

On Linux, the system doesn't always automatically switch rendering or compute between your internal GPU and your external one (especially when you connect/disconnect the eGPU). The `egpu-switcher` makes that switching clean and reliable.

In installed the latest [release tag 0.20.1](https://github.com/hertg/egpu-switcher/releases) after downloading the DEB package.

```bash
sudo cp Downloads/egpu-switcher-amd64 /opt/egpu-switcher
sudo chmod 755 /opt/egpu-switcher
sudo ln -s /opt/egpu-switcher /usr/bin/egpu-switcher
sudo egpu-switcher enable
```

### Step 4: Check the GPU

If we followed all the steps so far, `nvidia-smi` should be able to show the NVIDIA eGPU. If not, we should reboot with the eGPU connected and switched on.

```bash
# Single call
nvidia-smi

# Loop call for refeshed outputs
nvidia-smi -l 1
```

### Step 5: Install Docker with NVIDIA GPU Support

Containerization is essential for many applications. And what about containers with NVIDIA support? Follow this recipe to install [docker engine](https://www.docker.com/) with NVIDIA support.

```bash
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

# Test: We should get a hello world greeting
docker run hello-world
```

To **enable NVIDIA support**, we need to install 3 packages: `libnvidia-container`, `libnvidia-container-tools`, and `nvidia-container-toolkit`.
Their respective repositories can be found here:

- [https://github.com/NVIDIA/libnvidia-container/releases](https://github.com/NVIDIA/libnvidia-container/releases)
- [https://github.com/NVIDIA/nvidia-container-toolkit/releases](https://github.com/NVIDIA/nvidia-container-toolkit/releases)

In my case, the latest stable release was `v1.17.8`:

- [https://github.com/NVIDIA/nvidia-container-toolkit/releases/tag/v1.17.8](https://github.com/NVIDIA/nvidia-container-toolkit/releases/tag/v1.17.8)
- [https://github.com/NVIDIA/libnvidia-container/releases/tag/v1.17.8](https://github.com/NVIDIA/libnvidia-container/releases/tag/v1.17.8)

**However, I did not install them from that source.** Instead, I downloaded the compiled DEB packages from the following NVIDIA repository:

[https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/)

Here's the recipe I followed:

```bash
# We need to install 3 packages: libnvidia-container, libnvidia-container-tools, nvidia-container-toolkit
# Their repos are here:
#   https://github.com/NVIDIA/libnvidia-container/releases
#   https://github.com/NVIDIA/nvidia-container-toolkit/releases
# The latest version releases for me were:
#   https://github.com/NVIDIA/nvidia-container-toolkit/releases/tag/v1.17.8
#   https://github.com/NVIDIA/libnvidia-container/releases/tag/v1.17.8
# However, I couldn't find any readily compiled DEB packages there, just code
# and thanks to google I found this other official package repository:
#   https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/
# So I added the link to the repo list and then installed the packages automatically

# Add keys: 3bf863cc.pub is the official NVIDIA's signing key, located in the package repo
sudo apt update
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
  | sudo gpg --dearmor -o /etc/apt/keyrings/nvidia.gpg

# Tell Ubuntu here's a new package source that can be trusted with the above specific key
echo "deb [signed-by=/etc/apt/keyrings/nvidia.gpg] \
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

# Install packages
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Activate NVIDIA and restart docker daemon
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test GPU passthrough: We should see the output of nvidia-smi where our GPUs appear
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Step 6: Remote Access Configuration

I will be working with two machines:

- The Ubuntu machine (hostname `urgull`), which has a GPU attached to it.
- A Macbook Pro as the main UI (hostname `kasiopeia`), from which I would like to connect to the Ubuntu machine whenever I need to use the GPU.

In order to connect to our Ubuntu machine from the Mac, we need to

- get our local Ubuntu IP (and/or its hostname)
- set up the firewall on the Ubuntu machine
- and allow to connect to specific ports.

Here's how that's done:

```bash
# Get the IP of our Linux/Ubuntu - Local network
# We might get (0) the localhost 127.0.0.1, (1) the Ethernet IP, (2) the WLAN IP, and (3) the Docker IP
# We pick either the Ethernet or WLAN
ip -4 addr show | grep inet  # 192.168.x.x
# ... or
ip route | grep default
# ... or
hostname -I

# Check that the SSH server is installed and running
sudo apt install -y openssh-server
sudo systemctl enable --now ssh
sudo systemctl status ssh

# Setup firewall: enable/disable
sudo ufw enable  # ... to enable
sudo ufw disable  # ... to disable
sudo ufw reload

# If Firewall enabled: Allow SSH, SCP, VS Code Remote-SSH -- all port 22!
sudo ufw allow ssh
# If Firewall enabled: Allow Ollama port(s)
sudo ufw allow 11434/tcp
sudo ufw allow 11435/tcp
# Print status
sudo ufw status verbose
```

Now, from our Mac, we should be able to access our Ubuntu if we are in the same local network:

```bash
# Use your username & IP obtained before
ssh <username>@<ubuntu-ip>
ssh mikel@192.168.x.x
# Are you sure you want to connect? yes
```

We can also ssh to our local hostname; first, get the `hostname` on the Ubuntu machine:

```bash
hostname  # in my case, it's `urgull` 
```

Then, on the Mac, we can SSH to it as follows:

```bash
ssh <username>@<hostname>.local
ssh mikel@urgull.local
```

It makes sense to use the `hostname`, since the local IP might change over time. 

We can also register our Ubuntu machine in the `~/.ssh/config` of our Mac:

```bash
Host 192.168.x.x
  HostName 192.168.x.x
  User mikel

Host urgull.local
  HostName urgull.local
  User mikel
```

Note that no one from the Internet can connect to our Ubuntu machine unless we explicitly expose it.

- By default, no inbound connections from the Internet can reach internal IPs like `192.168.x.x`, because these are **local** IPs known only by our router.
- Our router is visible to the Internet with another **different public IP**.
- Our Ubuntu's SSH (port 22) and Ollama (port 11434) are not accessible from the Internet unless we manually tell our router to port-forward those ports.

## Using the GPU

If we follow all the steps in the previous section [Install Linux](#install-linux), we should have a GPU up and running on our machine; we can check that as follows:

```bash
# On Ubuntu machine with eGPU connected before boot
# Single call
nvidia-smi

# Loop call for refeshed outputs
nvidia-smi -l 1
```

In the following, I list some of the most common tool setups after the GPU installation:

- [Setting Up a GPU Python Environment](#setting-up-a-gpu-python-environment)
- [Quick Pytorch Example](#quick-pytorch-example)
- [Further Pytorch Tests in the Attached Notebook](#further-pytorch-tests-in-the-attached-notebook)

### Setting Up a GPU Python Environment

One of the easiest ways to handle Python environments in a data science context is Miniconda.

Here's how to [install Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install):

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

The environment installs mainly Pytorch (with GPU support) and some additional basic libraries.

I had some issues with `bitsandbytes`; re-installing it solved the issues:

```bash
conda activate gpu
pip uninstall bitsandbytes
pip install bitsandbytes
```

### Quick Pytorch Example

With the `gpu` environment, we can run this piece of code:

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

### Further Pytorch Tests in the Attached Notebook

The notebook [`test_gpu.ipynb`](./test_gpu.ipynb) contains some quick tests to showcase how to use the GPU:

- Basic Sequential Neural Network
- MNIST Training Test

I will be adding more examples, if I consider them interesting and I get some time ;)

## Further Useful Applications

### VSCode Server: Launch VSCode on Another Machine, but Hosted on the GPU-Ubuntu

Let's consider this scenario:

- we have our usual workstation, which is a Macbook (hostname `kasiopeia`)
- and we also have another machine with a GPU, i.e., the Ubuntu we have configured so far (hostname: `urgull`).

We'd like to work on the Macbook but use the powerful GPU from the Ubuntu.

Thanks to the VSCode [Remote-SSH Extension](https://code.visualstudio.com/docs/remote/ssh), that's very easy to achieve!

Pre-requisites: 

- Install the [Remote-SSH Extension](https://code.visualstudio.com/docs/remote/ssh). The section [Minimum VSCode Extensions](#minimum-vscode-extensions) below lists the extensions I frequently use.
- Set up the Ubuntu machine as explained above: sections [Install Linux](#install-linux) and [Using the GPU](#using-the-gpu).
- Register the GPU environment to appear in the Jupyter Kernel list, as shown in the following:

```bash
# On the Ubuntu machine

# Activate the GPU env
conda env list  # we should see, among others, the previously installed `gpu` env
conda init bash
conda activate gpu

# Register the current environment as a Jupyter kernel, to appear in the VSCode Kernel list
python -m ipykernel install --user --name=gpu --display-name "Python (gpu)"
```

After that, we can easily open a VSCode instance on our Macbook and connect to the Ubuntu from it:

- Click on the button *Open a Remote Window*, on the bottom left corner.
- Connect to a host...
- Enter: `<user>@<ubuntu-ip>` or, if configured `<user>@<ubuntu-hostname>.local`
- Enter Ubuntu user password
- ... et voilà! Our VSCode on the Macbook is actually running on Ubuntu.

Now, we can open the VSCode Terminal, which will launch on Ubuntu.

Alternatively, we can open the explorer view and select the folder we'd like to open.

If we open the current repository, we can run the notebook [`test_gpu.ipynb`](./test_gpu.ipynb) and we should get the GPU outputs as on the Ubuntu machine.

### Ollama Server: Use Ollama LLMs Running on the GPU-Ubuntu, but from Another Machine

Analogously to the VSCode use case, let's consider this scenario:

- we have our usual workstation, which is a Macbook (hostname `kasiopeia`)
- and we also have another machine with a GPU, i.e., the Ubuntu we have configured so far (hostname `urgull`).

Now, we'd like to work on the Macbook but use the powerful GPU from the Ubuntu to run an LLM.

To that end, assuming our Ubuntu is configured as explained in the section sections [Install Linux](#install-linux) and [Using the GPU](#using-the-gpu), we can install and test [Ollama](https://ollama.com/) on our Ubuntu:

```bash
# Install Ollama CLI and a the service ollama.service
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model to ~/.ollama/models
# The following models fit well with the RTX 3060
ollama pull llama3:8b
ollama pull mistral
ollama pull gemma:7b

# Get a list of available models
ollama list

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

# Show where the file is
# Usually
# /usr/share/ollama/.ollama/models
# or ~/.ollama/models
ollama show --file llama3:8b

# Force GPU usage
export OLLAMA_USE_GPU=1
ollama run llama3:8b

# If we want, we can start a/the service manually.
# If we have problems with the default port 11434,
# we can change it to 11435
export OLLAMA_HOST=127.0.0.1:11435
ollama serve &
# To stop:
ps -ef | grep ollama  # get PID
# ... or
sudo lsof -i :11434  # get PID
# ... and then stop it
kill -9 <PID>

# If Firewall enabled: Allow Ollama port(s)
sudo ufw allow 11434/tcp
sudo ufw allow 11435/tcp

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

Note that we have configured our firewall to allow connections on the Ollama ports.
To force the Ollama server to use the GPUs, we can set the `OLLAMA_USE_GPU` variable:

```bash
# Stop the Ollama service and start it again with GPU support
sudo systemctl stop ollama
export OLLAMA_USE_GPU=1
export OLLAMA_HOST=0.0.0.0:11434
ollama serve &

# If we have issues restarting the service
# we can fetch its PID and fill it
sudo lsof -i :11434
sudo kill -9 <PID>
# And the restart it
export OLLAMA_USE_GPU=1
export OLLAMA_HOST=0.0.0.0:11434
ollama serve &

# Optionally, if we have models in different places, we can use:
# export OLLAMA_MODELS=/usr/share/ollama/.ollama/models
# export OLLAMA_MODELS=~/.ollama/models
``` 

Then, we can use the Ollama API running on the Ubuntu machine from our Macbook:

```bash
# Get the models we have available
curl http://<ubuntu_ip>:11434/api/tags
curl http://<ubuntu-hostname>:11434/api/tags
curl http://urgull.local:11434/api/tags

# Run the generate API call
curl http://urgull.local:11434/api/generate -d '{
  "model": "llama3:8b",
  "prompt": "Write a haiku about machine learning."
}'
```

If we install Ollama on the Macbook, we can even use the Ollama service on the Mac, but letting the model run on the Ubuntu with the more powerful GPU!

```bash
# Install Ollama on the Macbook
curl -fsSL https://ollama.com/install.sh | sh

# We can even run the local Ollama on Macbook
# but using the Ubuntu server
export OLLAMA_HOST=urgull.local:11434
ollama run llama3:8b

# To revert to use the local Macbook Ollama service
export OLLAMA_HOST=127.0.0.1:11434
ollama run llama3:8b
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
  - iCloud

### Minimum VSCode Extensions

- **Remote-SSH**
- Python
- Jupyter
- DotENV
- Remote Development
- YAML
- LiveShare
- Github Copilot + Copilot Chat
- CMake Tools
- C/C++
- Markdown all in one
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

- [Enable Razer CoreX (NV-1070) eGPU on Razer Blade Stealth (NV-MX150) running Ubuntu 20.04](https://gist.github.com/tanmayyb/d19f9aa5641349f8830d05e2c91d5a79)
- [Guide for setting up e-gpu with framework 11th gen and Ubuntu 22.04](https://www.reddit.com/r/framework/comments/11dtm78/guide_for_setting_up_egpu_with_framework_11th_gen/)
- [Guide to install nvidia eGpu on ubuntu 22.04](https://gist.github.com/valteu/1c0a9b7288cc3d77a6654a4d22d0ce9f)
