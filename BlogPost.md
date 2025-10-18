# eGPU Blog Post

<!--
# Log in/out to Docker Hub
docker logout
docker login

# Pull the official image (first time)
docker pull excalidraw/excalidraw

# Start app
docker run --rm -dit --name excalidraw -p 5000:80 excalidraw/excalidraw:latest
# Open browser at http://localhost:5000

# Stop
docker stop excalidraw
docker rm excalidraw
docker ps
-->


![Ubuntu NVIDIA SMI](./assets/ubuntu_nvidia_smi.png)


[NVIDIA T4](https://www.nvidia.com/en-us/data-center/tesla-t4/)

[NVIDIA RTX 3060](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3060-3060ti/)


Title: My Personal eGPU Server Setup  
Subtitle: How to Run and Train LLMs Locally with NVIDIA Chips from a Mac & Linux 

You have propably followed the release of the [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) *personal supercomputer*. The device, with 128 GB of memory, 20 CPU cores, and a price of USD $3,999.00, will be definitely on the whish list of any AI nerd for this Christmas.

This blog post is my personal and humble alternative. Indeed, in the past 2 years I have been using a NVIDIA eGPU from my Macbook M1, but via a Linux machine, which plays the role of a server. Since some colleagues and friends showed interest, I decided to [**throughly document it on Github**](https://github.com/mxagar/linux_nvidia_egpu) and to write this blog post, which explains overall the setup and the motivation behind it. Here's the schematics of my *supercomputer*:

![eGPU Linux & Mac Setup](./assets/egpu_linux.png)

I mainly use the GPU to train general Deep Learning models (with [VSCode Remote Development](https://code.visualstudio.com/docs/remote/ssh)) and to run LLMs locally (with [Ollama](https://ollama.com/)); as you can see in the picture above:

- a
- b
- c

You might ask why I would want to run and train models locally, since we have many services available that spare us with the husle.








