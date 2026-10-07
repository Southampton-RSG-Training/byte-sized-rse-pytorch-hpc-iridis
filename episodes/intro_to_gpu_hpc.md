---
title: "Introduction to using GPUs on HPC Clusters"
teaching: 0 # teaching time in minutes
exercises: 0 # exercise time in minutes
---

::: questions

- How does GPU-based HPC differ from CPU-based HPC
- How can you use of a system with multiple GPUs effectively?

:::

::: objectives

-

:::

## HPC with GPUs

GPUs and CPUs are both designed to perform computations, but have architectures which are optimized differently:

- modern CPUs can be thought of as narrow but deep: they can only do a few things simultaneously—generally just one thing per core—but they heavily optimize that work, by doing things like looking ahead in the instruction pipeline and pre-computing based on what they suspect the most likely code path is. They are fast, but trade off performance for generality.
- modern GPUs are wide, but shallow: they can work simultaneously on large amounts of date, but the operations being performed have to be the same. They evolved from needing to perform fast linear algebra computations on pixels in a buffer for display on a screen, but this also turned out to be useful for other linear algebra computations, and over the past 20 years they have increasingly been used for neural networks and deep learning which also heavily rely on performing linear algebra in parallel on multiple inputs. Each GPU core is doing the same operations over more than one value at a time. The amount of data that they can work on is constrained by the GPU memory.

So GPUs excel in HPC tasks where you have data which fits into GPU memory but which you wish to apply the same operations to all of it. They don't work as well with extremely large data, very small data, or where the computations that need to be performed are very different for different inputs.

But all of this means that they are very well suited for deep neural networks trained using back-propagation and gradient descent algorithms.

The GPU on a personal computer typically has a handful of cores and a few gigabytes of GPU memory. This limits the sorts of computations in a number of ways:

- only a handful of different computational steps can be performed at the same time
- the working data is comparatively small
- when the working data is larger than the available GPU memory, or when you switch computational tasks, you often need to spend time moving data between GPU and main memory (or worse, to disk or network).

Just like HPC systems based around CPUs, GPU-base HPC systems typically offer more GPU cores (often in the thousands, with specialized cores for particular data types), more GPU memory (into the 10s of gigabytes), and faster access to that memory. They may also have multiple GPUs installed on the same machine.

## Programming with GPUs

GPUs have different instruction sets than CPUs, and the code that they run tends to be small and simple. Rather than loading pre-compiled code from disk, as a CPU-based program might do, it's common for code that is using a GPU to compile that code on-the-fly for the particular GPU that is going to be used, allowing it to make greatest use of the hardware.

In the early days, this was done via mechanisms like OpenGL's shader language, then OpenCL's compute kernels, but more recently more hardware-specific technologies have prevailed: CUDA for Nvidia hardware, ROCm for AMD hardware, and MPS for Apple Silicon.

If you know the hardware which your code is going to run on, you can work at this level writing code in C or C++, or using wrappers in Python or another high-level language.

However, it's more common to use more domain-specific code which uses different GPU backends as appropriate for the hardware being used, but which can also run on the CPU if nothing else is available.

In particular, libraries like TensorFlow, Torch and JAX provide unified tensor APIs for machine learning, handling the automatic differentiation needed for back-propagation and gradient-descent based algorithms.

Libraries like Keras and Transformers then sit on top of those layers to provide deep learning and LLM-specific support.

One complication that filters up to the level of the person writing code using GPUs, even at a high level, is that GPUs usually have their own dedicated memory hardware, and so programmers need to pay attention to *where* the data and tensor weights are stored: in main memory or GPU memory (and if there are multiple GPUs, *which* GPU), as that affects where computation can take place; and trying to perform a computation where the data is on different devices will likely result in a program crash.

## Multi-GPU code

When data or models become large enough, speed-up can potentially be gained by running on multiple GPUs or servers in parallel. There are two potential ways of doing this:

- **Data Parallel** computations run the same computation, but split the data across GPUs and/or servers. For example, splitting data across multiple GPUs when performing inference using a deep learning model.

- **Model Parallel** computations split the model across different GPUs and run the data through all of them. For example, splitting model layers across GPUs when performing inference using a deep learning model. For very large models, a large single layer may even need to be spread over multiple GPUs.

For deep learning models, particularly when training, care has to be taken when synchronizing between the different GPUs. For example, when training, the code may need to carefully synchronise steps. For example the results of a layer computation on a batch in a model parallel example may need to wait for the next layer to finish the previous batch before data is transferred to the next GPU.

## Deep Learning Libraries

The major libraries for working with deep learning models are Torch, TensorFlow and JAX. These are written in C++, but all have Python bindings that integrate efficiently with the standard Python scientific libraries, particularly NumPy and PIL.

In 2023 roughly 60% of AI research papers and over 90% of published models on HuggingFace use PyTorch. However most of the popular models were available for both TensorFlow and PyTorch. JAX is comparatively new, so doesn't yet have the mind-share of other libraries, but is backed by Google.

Keras is available as an API which unifies all three libraries, allowing the use of fairly similar code for building and training standard neural network architectures. Since it focuses on the networks and standardizes the training, it is a little harder to run across multiple GPUs or servers.

We'll primarily use PyTorch directly in this session, since it is the most popular framework, and also because the tools for multi-gpu deployment are somewhat nicer.

## Multi-GPU PyTorch

PyTorch has a number of different APIs for distributing computation across multiple GPUs, or handling situations where the model itself is too large to fit within a single GPU.

- **DistributedDataParallel (DDP)** is for use when your model fits within a single GPU, but you want to scale processing of data across multiple GPUs.
- **FullyShardedDataParallel (FSDP2)** is for when your model does not fit into one GPU.
- **Tensor Parallel (TP)** allow scaling beyond what FSDP2 is capable of.
- there are several other APIs, such as DeviceMesh, Monarch and TorchTitan.

These APIs are complex, and it is easy to miss subtle requirements for synchronisation, and not easy to convert from one API to another. Code written for DDP, for example, will need re-work to run using FSDP2. That makes the development process going from simple working CPU or single-GPU code to fully distributed code fragile.

HuggingFace has a library, Accelerate, which simplifies the process of switching between the different APIs, by providing a unified API that can be configured to work with whatever hardware you are running on. The approach that Accelerate takes, is to wrap non-distributed PyTorch API objects, which then ensures that data and models get spread around the declared resources appropriately. For research software, as opposed to commercial production pipelines, Accelerate provides a practical solution to most of the difficulties of distributing GPU computation.

Hugging-face

## Checkpoints

When training models, particularly when you expect the training to take a long time, you want to be able to inspect progress and performance of the models. Additionally, if your code runs into difficulties, or ends up running long, you do not want to lose progress you have made up to this point.

Some of this can be achieved by running validation tests at the end of each epoch of training to measure loss and other metrics against non-training data.

But better is to periodically save the state of your model, so that you can resume training from any point if you need to, and you can use the partially trained models to validate your approach, and potentially cancel the job early if you already have what you need, or if there is a difficulty with the way you are doing things.

This is known as check-pointing.

Distributed computation makes check-pointing somewhat more complex, but again, Accelerate has methods that let you checkpoint distributed training easily.

## Iridis X and Iridis 7

At the University of Southampton there are two clusters which provide access to HPC GPU computation: Iridis X and the new Iridis 7.

Iridis X consists machines with a total of 46 Nvidia A100 80GB graphics cards, and an additional 16 H100 80GB graphics cards and 20 H200 142GB graphics cards (some of which are reserved for ECS or Mathematical Sciences research). A typical machine has 2, 4 or 8 graphics cards, 500GB of main memory, and 48 CPU cores.

Iridis 7, which is very new, adds another 120 H200 graphics cards.

They share disk storage with Iridis 6, but have different login nodes (which are currently shared by Iridis X and 7). They also shared the login accounts: You don't need separate accounts for each of them.

Two of the Iridis X login nodes have a GPU available which can be used for quick validation of code before submitting something more substantial.

### SLURM Configuration for Using GPUs

When using a GPU cluster for your work, you need to specify to SLURM the resources you expect to use. In addition to specifying the number of nodes and processes, you also need to specify how many GPUs you need per process. GPU clusters typically have restrictions on how many nodes you can use compared to CPU nodes.

You can do this either by specifying the number of GPUs to use per node:

``` bash
#SBATCH --gpus=2
```

or by the number of GPUs per task:

``` bash
#SBATCH --gpus-per-task=2
```

Obviously, these values must be within the number available with the hardware, and they must be within the limits of what you can access with your usage rights.

Depending on the technology being used to transfer data between GPUs, you may also need to specify:

``` bash
#SBATCH --gpu-bind=none
```

as GPU binding can interfere with native inter-GPU communication.

Also, you want to make CUDA available so you can run your code on the GPUs. You do this by loading the `cuda` module the same way that you load the `python` module:

``` bash
load module cuda
```


## References

[AssemblyAI, PyTorch vs TensorFlow in 2023](https://www.assemblyai.com/blog/pytorch-vs-tensorflow-in-2023)

[PyTorch Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)
