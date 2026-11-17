class: title

# Byte-sized RSE: Training Deep Learning Models with HPC
## Corran Webster
### Southampton Research Software Group

---
class: title

# Overview

**Introduction:** Training Deep Learning Models with HPC

**Practical activity:** Multi-GPU Training

---
class: title reverse

# Training Deep Learning Models with HPC
# Introduction

---

# GPU-based HPC and Iridis

- Some HPC clusters offer nodes with high-performance GPUs.
- The University of Southampton offers these as part of the Iridis X and Iridis 7 clusters.
- Access is the same as for Iridis 6, but through the `loginX001`, `loginX002`, and `loginX003` nodes for both X and 7. Storage is shared with Iridis 6.
- The GPU nodes are in comparatively high demand, and usage is generally restricted to a node or two.
- There are many different partitions with different characteristics.

???

-

---

# Deep Learning Libraries

- The most common application for GPU-based HPC is working with deep learning models.
- Common libraries for this include TensorFlow, Torch and JAX, all of which are written in C++ with Python bindings. Torch is the most popular within the research world.
- The key features of these libraries are Tensor objects with support for automatic differentiation, allowing fast forward computation and efficient back-propagation.

---

# Developing GPU Research Code

- All of these libraries allow CPU-based computation alongside GPU-based acceleration with CUDA or MPS.
- This permits local development on personal computers or workstations.
- However all of these libraries need to pay attention to the hardware, requiring some effort to make the same code run cleanly both locally and on a cluster.
- Additionally GPUs generally have more limited resources than CPUs and this needs to be taken into account when devising algorithms.

---

# Checkpoints and Logging

- For all but the simplest of computational code, you want to allow for the possibility that your code is interrupted.
- This can be done by saving sufficient intermediate state that your code can resume from the last known-good point.
- For deep learning, this includes not just model state, but optimizers and learning rate scheduling as well.
- Similarly, good logging allows you to evaluate progress of your code and diagnose failures.

---

# Multi-GPU Approaches

- One solution to the constraints of running on a single GPU is to spread the workload over multiple GPUs.
- Data parallel approaches replicate the full deep learning model on each GPU but run data through in parallel.
- Other, "sharded" approaches put different layers on different devices or even split large layers across multiple GPUs.
- These approaches add complexity because data and weights must be synchronized between the various devices.

---

# SLURM and GPUs

- you need `module load cuda`
- when scheduling jobs that use GPUs you need to specify:
  - which partition you want (and so the *type* of GPU)
  - how many GPUs you want
  - how many CPUs you want
- With Accelerate use one SLURM Task per Node, each Task using at least as many CPU cores as GPUs.
- Use as many GPUs on a single node as you can before using multiple nodes.

---

# Example Batch File

.left-column[

```bash
#!/bin/bash

#SBATCH --partition=l4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=4

module load cuda
module load python

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

accelerate launch --multi-gpu \
  --num_processes=4 my_script.py

```
]
.right-column[

- This will run a 4 GPU task on one node.
- Accelerate will use the 4 processes to talk to each of the GPUs.

]

???

---

# Modules

- Different HPC programs use different versions of underlying technologies
- You need to select which technologies you are using, such as languages, GPU support, compilers, etc. through the **module** system on Iridis
- For example, if your code needs Python 3.14 you load it using the Bash command `module load python/3.14`
- You can see all available modules with the `module avail` command.

---

# Parallel Tasks

- if you have multiple tasks the batch script only runs in the main task. To execute in parallel in multiple tasks you need to prefix your main command with `srun`
- you can inspect environment variables to see which task and node the code is running
- in Python use `os.environ.get("SLURM_PROCID")` to get the index of your task.
- other useful environment variables include `SLURM_NODEID`, `SLURM_LOCALID`, and `SLURM_NTASKS`.

---

```bash
#!/bin/bash

#SBATCH --job-name=python-example
#SBATCH --partition=batch
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1

# Load required modules
module load python

# Activate Python virtual environment
source ~/myenv/bin/activate

# Set any environment variables or configuration options
export PYTHONUNBUFFERED=1

# Move to job directory
cd $SLURM_SUBMIT_DIR

# Run the Python script on multiple nodes
srun python my_script.py
```

---

# Effective Use of GPUs

- You want to develop as much as you can locally, on CPU if needed.
- Set up check-point and logging code.
- Estimate your memory and GPU needs.
- Select the appropriate partition based on that.

---

class: title reverse

# Simple CNN Training
# Practical Activity

---

# Activity Introduction

- We have code that trains a simple CNN model.
- We'll demonstrate this via live coding.
  - We'll use a repository with example code.

---
class: title reverse

# Thank you!
# What questions do you have?
