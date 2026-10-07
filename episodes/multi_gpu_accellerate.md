---
title: "Multi-GPU Deep Learning with Accelerate"
teaching: 0 # teaching time in minutes
exercises: 0 # exercise time in minutes
---

::: questions

- How can you take advantage of GPU acceleration on Iridis X and 7?
- How can you use of a system with multiple GPUs effectively?

:::

::: objectives

- Adapt a PyTorch training script to use Huggingface Accelerate.
- Run PyTorch code on a single GPU on an Iridis X node.
- Distribute training of a deep learning learning classification model across multiple GPUs on a single machine.

:::

## Activity - Train a simple CNN model on Iridis X

### The model

The starting point is a simple convolutional neural network designed for classification using the MNIST digits data set.  This code is taken from the PyTorch examples.

``` python
from torch.nn import Module, Conv2d, Dropout, Linear
from torch.nn.functional import relu, max_pool2d, log_softmax

class SimpleCNN(Module):
    """CNN for MNIST Data

    From PyTorch examples: https://github.com/pytorch/examples/blob/main/mnist/main.py
    """

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 32, 3, 1)
        self.conv2 = Conv2d(32, 64, 3, 1)
        self.dropout1 = Dropout(0.25)
        self.dropout2 = Dropout(0.5)
        self.fc1 = Linear(9216, 128)
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = relu(x)
        x = max_pool2d(x, 2)
        x = self.dropout1(x)
        x = flatten(x, 1)
        x = self.fc1(x)
        x = relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = log_softmax(x, dim=1)
        return output
```

We can train this model for one epoch with a function like this:

``` python
def train_epoch(
    model,
    train_loader,
    optimizer,
):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = nll_loss(output, target)
        loss.backward()
        optimizer.step()
```

And then that can be put into a loop that trains epoch-by-epoch:

``` python
def train(
    model,
    optimizer,
    train_loader,
    epochs,
    gamma=0.7,
):
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer)
        scheduler.step()
```

Inference can then be done with a function like this:

``` python
def predict(model, data):
    model.eval()
    with no_grad():
        return model(data)
```

And testing and validation can be done with a function like:

``` python
def test(model, test_loader):
    test_loss = 0
    correct = 0
    with no_grad():
        for data, target in test_loader:
            output = predict(model, data)
            test_loss += nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    return test_loss, correct
```

Best practices would encourage adding things like:

- logging of progress
- validation testing
- checkpoints and the ability resume testing
- a command-line interface to simplify use

Adding all that in gives code which looks like this:

``` python
from math import inf
import logging

import torch
from torch import flatten, no_grad
from torch.nn import Module, Conv2d, Dropout, Linear
from torch.nn.functional import relu, max_pool2d, log_softmax, nll_loss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, random_split

logger = logging.getLogger(__name__)


class SimpleCNN(Module):
    """CNN for MNIST Data

    From PyTorch examples: https://github.com/pytorch/examples/blob/main/mnist/main.py
    """

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 32, 3, 1)
        self.conv2 = Conv2d(32, 64, 3, 1)
        self.dropout1 = Dropout(0.25)
        self.dropout2 = Dropout(0.5)
        self.fc1 = Linear(9216, 128)
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = relu(x)
        x = max_pool2d(x, 2)
        x = self.dropout1(x)
        x = flatten(x, 1)
        x = self.fc1(x)
        x = relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = log_softmax(x, dim=1)
        return output


def train_epoch(
    model,
    train_loader,
    optimizer,
    epoch,
    log_interval=10,
):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            # report progress
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0%})]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def predict(model, data):
    model.eval()
    with no_grad():
        # data = data.to(device)
        return model(data)


def test(model, test_loader):
    test_loss = 0
    correct = 0
    with no_grad():
        for data, target in test_loader:
            output = predict(model, data)
            # target = target.to(device)
            test_loss += nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0%})\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            correct / len(test_loader.dataset),
        )
    )
    return test_loss


def train(
    model,
    optimizer,
    train_loader,
    validation_loader,
    epochs,
    checkpoint_dir,
    start=1,
    patience=None,
    gamma=0.7,
    log_interval=10,
):
    best_val_loss = inf
    if patience is None:
        patience = epochs
    wait = 0
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    if start is None:
        # try to resume if there are any checkpoints
        checkpoints = sorted(checkpoint_dir.glob("SimpleCNN_*"))
        if checkpoints:
            start = load(model, optimizer, scheduler, checkpoints[-1]) + 1
        else:
            start = 1

    for epoch in range(start, start + epochs):
        train_epoch(
            model, train_loader, optimizer, epoch, log_interval
        )
        val_loss = test(model, validation_loader)
        scheduler.step()
        if val_loss < best_val_loss:
            # checkpoint if model is better than previous best
            save(model, optimizer, scheduler, epoch, checkpoint_dir)
            wait = 0
        else:
            wait += 1
            if wait > patience:
                logger.info(f"Training stabilised at epoch {epoch}.")
                break


def save(model, optimizer, scheduler, epoch, path):
    path.mkdir(parents=True, exist_ok=True)
    filename = f"SimpleCNN_{epoch:0>5d}"
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        },
        path / filename
    )
    logger.info(f"Checkpoint saved: {path}/{filename}")


def load(model, optimizer, scheduler, path):
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = int(path.name.split("_")[-1])
    return epoch
```

And a little `click`-based command-line utility to run everything, including a preprocess command that downloads the data set if it isn't already available.

``` python
import logging
from pathlib import Path
import os

import click

@click.group()
@click.option(
    "--log-dir",
    "log_dir_name",
    type=click.Path(file_okay=False, writable=True),
    default="",
)
def cli(log_dir_name):
    if log_dir_name:
        # ensure logging directory exists
        log_dir = Path(log_dir_name)
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=log_dir / f"log-{os.getpid()}.log",
            level=logging.INFO,
        )
    else:
        logging.basicConfig(level=logging.INFO)


@cli.command()
@click.option(
    "--data-dir",
    "data_dir_name",
    type=click.Path(file_okay=False, writable=True),
    default="data",
)
def preprocess(data_dir_name: str):
    """Perform preprocessing commands before distributing to nodes."""
    from .data import download_mnist_data

    # ensure data directory exists
    data_dir = Path(data_dir_name)
    data_dir.mkdir(parents=True, exist_ok=True)

    download_mnist_data(data_dir)


@cli.command()
@click.option(
    "--data-dir",
    "data_dir_name",
    type=click.Path(exists=True, file_okay=False),
    default="data",
)
@click.option(
    "--checkpoints-dir",
    "checkpoints_dir_name",
    type=click.Path(file_okay=False, writable=True),
    default="checkpoints",
)
@click.option("--split", default=0.8)
@click.option("--epochs", default=10)
@click.option("--resume/--no-resume")
def train(
    data_dir_name: str,
    checkpoints_dir_name: str,
    split: float,
    epochs: int,
    resume: bool,
):
    """Train on the downloaded data."""
    try:
        from torch.optim import Adadelta
        from torch.utils.data import DataLoader, random_split

        from .model import SimpleCNN, train
        from .data import load_mnist_data

        data_dir = Path(data_dir_name)
        checkpoints = Path(checkpoints_dir_name)
        if resume:
            start = None
        else:
            start = 1
        # ensure checkpoints directory exists
        checkpoints.mkdir(parents=True, exist_ok=True)


        model = SimpleCNN()
        optimizer = Adadelta(model.parameters())

        dataset = load_mnist_data(data_dir)
        training_data, validation_data = random_split(dataset, [split, 1 - split])
        training_loader = DataLoader(training_data, batch_size=64)
        validation_loader = DataLoader(validation_data, batch_size=64)

        train(
            model,
            optimizer,
            training_loader,
            validation_loader,
            epochs,
            checkpoints,
            start=start,
        )
    except Exception:
        logger = logging.getLogger(__name__)
        logger.exception("Exception during training:")
```

This runs training on the CPU only.  Our end goal is to run the training code on Iridis X using the available GPUs with CUDA. To do this, in addition to the set-up to run things using SLURM, we need to adapt the code to be able to run on CUDA GPUs.

Developing code using Iridis is awkward: you can't use a graphical IDE, the test cycle is slow, and the login nodes are often busy. So we want to do as much work as we can locally on your personal machine.

::: challenge

### Run the code on your machine.

Use the "Template" command on the [GitHub repository](https://github.com/Southampton-RSG-Training/byte-sized-rse-pytorch-hpc-example) to create a copy of the code in your Github account. For simplicity, make this public.

Clone the code to your local machine with `git` using:

``` bash
cd
git clone git@github.com:[YOUR_GITHUB_USERNAME]/byte-sized-rse-pytorch-hpc-example.git
```

and then from your local Bash shell, run:

``` bash
cd byte-sized-rse-pytorch-hpc-example
./install_local.sh
source venv/bin/activate
```

which sets up a local virtual environment with the code installed and downloads the data. You can then try running the code with:

``` bash
simple-cnn train --epochs 2
```

::: solution

If you run the code successfully, it should look like:

```
INFO:simple_cnn.model:Train Epoch: 1 [0/48001 (0%)]	Loss: 2.307507
INFO:simple_cnn.model:Train Epoch: 1 [640/48001 (1%)]	Loss: 1.586141
INFO:simple_cnn.model:Train Epoch: 1 [1280/48001 (3%)]	Loss: 0.761963
INFO:simple_cnn.model:Train Epoch: 1 [1920/48001 (4%)]	Loss: 0.461516
...
INFO:simple_cnn.model:Train Epoch: 2 [46080/48001 (96%)]	Loss: 0.043385
INFO:simple_cnn.model:Train Epoch: 2 [46720/48001 (97%)]	Loss: 0.061486
INFO:simple_cnn.model:Train Epoch: 2 [47360/48001 (99%)]	Loss: 0.034389
INFO:simple_cnn.model:Train Epoch: 2 [750/48001 (100%)]	Loss: 0.000566
INFO:simple_cnn.model:Test set: Average loss: 0.0464, Accuracy: 11839/11999 (99%)
```

:::

:::

## Generalizing code using Accelerate

The first task is to generalize the code so that it runs using a GPU if it's available. Usually this involves code which detects the system and attempts to work out what the appropriate PyTorch `device` value is: `"cuda"` on an Nvidia GPU, `"mps"` on an Apple Silicon mac, etc. but also keep it working with just the CPU if run on a machine without an appropriate GPU. You then need to make sure that training and inference run on the correct device, primarily by using the `.to(device)` methods to make sure that the weight and data tensors are on the correct devices.

This is moderately complex, and it can be hard to test that you have distributed things correctly for an Iridis X node, particularly if you are using a machine which only has CPU support from PyTorch.

The solution which we are going to use is the [Accelerate](https://huggingface.co/docs/accelerate/index) library by HuggingFace. It not only contains the code to detect the available devices and use them appropriately, it also wraps `torch.distributed` (which in turn wraps a number of distribution technologies, including MPI) to permit the work to be distributed, and simplifies code for logging and loading/saving training state.

The changes to make your code do this are simple:

1.  In the training code import `Accelerate` from the `accelerate` library and create an instance of the `Accelerate` class:

    ``` python
    from accelerate import Accelerate

    accelerator = Accelerate()
    ```

2.  Prepare the key objects involved in the training by calling `accelerator.prepare` with them:

    ``` python
    model, optimizer, training_loader, scheduler = accelerator.prepare(
        model, optimizer, training_loader, scheduler
    )
    ```

    The `prepare` call expects a collection of objects to prepare, but the order doesn't matter; but it *does* return the prepared objects in the same order, so the inputs must match the outputs.

3.  Instead of using `loss.backward()` to do back-propagation, instead use `accelerator.backward(loss)`.

4.  Remove any calls of the `.to(device)` method: moving data to the appropriate device is handled automatically.

5.  Run the code with `accelerate launch` instead of `python`:

    ``` bash
    accelerate launch -m simple_cnn train --epochs 2
    ```

The `accelerate launch` command is technically optional, but it can take additional command-line arguments that give you control over the way the code is run. For example, to run the code on the CPU (even if there is a GPU), you can pass `--cpu` as a command-line argument:

``` bash
accelerate launch --cpu -m simple_cnn train --epochs 2
```

::: challenge

### Convert the Training Code to use Accelerate

Change the code to make use of Accelerate and then run it on your machine.

Add, commit and push the changes to your copy of the github repo.

::: solution

The updated training code should look like:

``` python
```

And run with:

``` bash
accelerate launch -m simple_cnn train --epochs 2
```

:::

:::

### Checkpoints with Accelerate

When running training in batch mode it's wise to save checkpoints of the state of the training occasionally. The code in the example uses PyTorch's `save` and `load` systems to save the state of the objects involved in training:

``` python

def save(model, optimizer, scheduler, epoch, path):
    path.mkdir(parents=True, exist_ok=True)
    filename = f"SimpleCNN_{epoch:0>5d}"
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        },
        path / filename
    )
    logger.info(f"Checkpoint saved: {path}/{filename}")


def load(model, optimizer, scheduler, path):
    epoch = int(path.name.split("_")[-1])
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return epoch
```

These have to save more than just the current state of the model, but all objects involved in the training.

Accelerate allows us to simplify this, because it has already been told what objects are involved in the training. The `save_state` and `load_state` methods handle serializing and deserializing everything in one operation.

So we can replace `torch.save` by `accelerator.save_state`, and `torch.load` with `accelerator.load_state`:

``` python
def save(accelerator, epoch, path):
    path.mkdir(parents=True, exist_ok=True)
    filename = f"SimpleCNN_{epoch:0>5d}"
    accelerator.save_state(path / filename)
    logger.info(f"Checkpoint saved: {path}/{filename}")


def load(accelerator, path):
    epoch = int(path.name.split("_")[-1])
    accelerator.load_state(path)
    return epoch
```

Then we need to adjust the calls to `save` and `load` to pass the accelerator rather than the `model`, `optimizer`, and `scheduler`. The resulting code is simpler and more robust than the original code.

::: challenge

### Convert the Checkpoint Code to use Accelerate

Clear the checkpoint directory with the command:

``` bash
rm checkpoint/SimpleCNN*
```

Change the code to make use of Accelerate for making checkpoints and then run it on your machine.

You can verify that the code works by running the test suite:

``` bash
python -m unittest discover --s src -v
```

Add, commit and push the changes to your copy of the github repo.

::: solution

The updated training code should look like:

``` python
```

And run with:

``` bash
accelerate launch -m simple_cnn train --epochs 2
```

:::

:::

### Convert Logging to us Accelerate

As written, the code currently uses Python's standard logging system to save logs. Logging best practice in Python is to import `logging` in every module where you wish to do logging, and then call `logging.getLogger(__name__)` to set up the module's logger object:

``` python
import logging

logger = logging.getLogger(__name__)
```

Then `logger` is available as a module-level global to perform any logging activities. For example, we can log the state of training in every nth batch as follows:

``` python
def train_epoch(
    model,
    train_loader,
    optimizer,
    epoch,
    log_interval=10,
):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            # report progress
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0%})]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    batch_idx / len(train_loader),
                    loss.item(),
                )
            )
```

The other task that needs to be done at the start of each process is to configure logging. This is done in the example in the CLI:

``` python
import logging

@click.group()
@click.option(
    "--log-dir",
    "log_dir_name",
    type=click.Path(file_okay=False, writable=True),
    default="",
)
def cli(log_dir_name):
    if log_dir_name:
        # ensure logging directory exists
        log_dir = Path(log_dir_name)
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=log_dir / f"log-{os.getpid()}.log",
            level=logging.INFO,
        )
    else:
        logging.basicConfig(level=logging.INFO)
```

This either logs to standard output, or to a specified directory (using the process id in the log file name).

::: challenge

### Convert the Checkpoint Code to use Accelerate

Clear the checkpoint directory with the command:

``` bash
rm checkpoint/SimpleCNN*
```

Change the code to make use of Accelerate for making checkpoints and then run it on your machine.

You can verify that the code works by running the test suite:

``` bash
python -m unittest discover --s src -v
```

Add, commit and push the changes to your copy of the github repo.

::: solution

The updated training code should look like:

``` python
```

And run with:

``` bash
accelerate launch -m simple_cnn train --epochs 2
```

:::

:::

This doesn't work as well when running multiple tasks, and particularly on multiple nodes, as each Python process gets its own standard output and file, so it can be hard to tell what things happen when on different processes and nodes.

Accelerate helps with this by providing an almost drop-in replacement for the `logging` module that sends all logging back to a single node. To use it, you replace `import logging` with `from accelerate import logging` and `getLogger` with `get_logger`. Everything else remains unchanged:

``` python
from accelerate import logging

logger = logging.get_logger(__name__)
```

::: challenge

### Update Logging for Accelerate

Update the logging code using Accelerate as described above, and then verify that the code runs correctly using the tests.

``` bash
python -m unittest discover --s src -v
```

Add, commit and push the changes to your copy of the github repo.

:::

::: spoiler

### Uploading data to Iridis with `scp`

In these examples we've downloaded the data from the internet. But what if your data is on *your* computer?

In this case the `scp` command is likely what you want. It is usually installed as part of SSH, and uses SSH to securely transfer files between computers.

:::

### Running with SLURM

The basics of running your code with SLURM are similar to running on a CPU cluster, with the additional need to specify the number of GPUs to use. A minimal sbatch file to run on one node and one GPU might look something like:

``` bash
#!/bin/bash

#SBATCH --job-name=simple-cnn-example
#SBATCH --partition=a100
#SBATCH --time=00:05:00
#SBATCH --nodes=1                         # Number of Nodes (max=1)
#SBATCH --ntasks=1                        # Number of Tasks to spawn
#SBATCH --gpus-per-task=1                 # Every task to use one GPU
#SBATCH --cpus-per-task=1                 # Accelerate spawns a worker per GPU
#SBATCH --gpu-bind=none                   # NCCL can't deal with task-binding

WORKING_DIRECTORY=simple_cnn_workspace
VENV_NAME=$WORKING_DIRECTORY/venv
DATA_DIR=$WORKING_DIRECTORY/data
CHECKPOINTS_DIR=$WORKING_DIRECTORY/checkpoints
LOG_DIR=$WORKING_DIRECTORY/logs

# Optional: print useful job info
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "Number of GPUs: $SLURM_GPUS_PER_TASK"

# Load required modules
module purge
module load python/3.14
module load cuda

# Activate Python virtual environment
source $VENV_NAME/bin/activate

# Set any environment variables or configuration options
export PYTHONUNBUFFERED=1

# Move to job directory
cd $SLURM_SUBMIT_DIR

# Run the Python script
accelerate launch --num_processes 1 -m simple_cnn --log-dir $LOG_DIR train --data-dir $DATA_DIR --checkpoints-dir $CHECKPOINTS_DIR --epochs=2 --resume

# deactivate virtual environment
deactivate
```

The key features are:

- one node per task; in this session we are only considering single node jobs.

  ``` bash
  #SBATCH --nodes=1                         # Number of Nodes (max=1)
  #SBATCH --ntasks=1                        # Number of Tasks to spawn
  ```

- the number of CPUs per task should be more than the number of GPUs (equal is fine in most cases), and the number of GPUs per task is limited by the number of GPUs per node

  ``` bash
  #SBATCH --gpus-per-task=1                 # Every task to use one GPU
  #SBATCH --cpus-per-task=1                 # Accelerate spawns a worker per GPU
  ```

- use CUDA:

  ``` bash
  module load cuda
  ```

- run accelerate with a number of processes equal to the number of GPUs per node:

  ``` bash
  accelerate launch --num_processes 1 ...
  ```

::: challenge

### Run with Multiple GPUs on One Node

Update the SLURM script to run on 2 GPUs (but still only 1 node).

Add, commit and push the changes to your copy of the github repo.

::: solution

You should update the number of GPUs and CPUs per task to 2:

``` bash
#SBATCH --gpus-per-task=2                 # Every task to use two GPUs
#SBATCH --cpus-per-task=2                 # Accelerate spawns a worker per GPU
```

You should als run accelerate with 2 processes:

``` bash
accelerate launch --num_processes 2 ...
```
:::

:::


### Running on Iridis X

Once our code is running properly locally with accelerate, we can run it on Iridis. We need to get the source code onto Iridis, which we can do either by cloning the GitHub repo or copying it with a tool like `scp` or `rsync`. Generally using GitHub is best from a reproducibility standpoint, because you are running code which has a known state.

If you don't intend to modify the code, and your repo is public, you can clone the repo over HTTPS, but if you are planning to push changes back to the repo, or it is private, you will need to set up your GitHub SSH credentials and connect that to your GitHub account.

``` bash
git clone https://github.com/Southampton-RSG-Training/byte-sized-rse-pytorch-hpc-example.git
```

Either way, once you have the cloned repository you will need to set up the system and download the training data from the login node. The `install.sh` Bash script will do this for you.

``` bash
cd byte-sized-rse-pytorch-hpc-example
module load python
./install.sh
```

To quickly test that everything is working on Iridis you can run the tests:

``` bash

```

## References

[PyTorch MNIST Example](https://github.com/pytorch/examples/blob/main/mnist/main.py)

[Python Logging How-To](https://docs.python.org/3/howto/logging.html)

::: keypoints


:::
