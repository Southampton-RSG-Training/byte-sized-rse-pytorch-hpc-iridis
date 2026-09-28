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

- Run PyTorch code on a single GPU on an Iridis X node.
- Adapt a PyTorch training
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

We can train this model with the CPU with a function like this:

``` python
from torch.optim import Adadelta
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from .model import SimpleCNN

def train(epochs, gamma=0.7):
    device = 'cpu'
    model = SimpleCNN().to(device)

    optimizer = Adadelta(model.parameters())

    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    dataset = MNIST(transform=transform, train=True)
    training_data, validation_data = random_split(dataset, [split, 1 - split])

    training_loader = DataLoader(training_data, batch_size=64)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = nll_loss(output, target)

            loss.backward()
            optimizer.step()

        scheduler.step()

    return model
```

And we have a little `click`-based command-line utility to run everything.

``` python
```

Our goal is to run the training code on Iridis X.  But we need to do some work to prepare for this and it is much easier to do it on your personal computer.

::: challenge

### Run the code on your machine.

Use the "Template" command on the ... GitHub repository to create a copy of the code in your Github account.

Clone the code to your local machine with `git` from ..., set up a virtual environment, and install the dependencies with `pip install -e .`

::: solution

The commands should look like:
``` bash
git clone ...
cd ...
python3 -m venv venv
source venv/bin/activate
pip install -e .
python -m simple_cnn train --epochs 2
```

:::

:::

## Generalizing code using Accelerate

The first task is to generalize the code so that it runs using a GPU if it's available. Usually this involves code which detects the system and attempts to work out what the appropriate PyTorch `device` value is: `"cuda"` on an Nvidia GPU, `"mps"` on an Apple Silicon mac, etc. but also keep it working with just the CPU if run on a machine without an appropriate GPU.

The [Accelerate](https://huggingface.co/docs/accelerate/index) library by HuggingFace does this and more. It not only contains the code to detect the available devices and use them appropriately, it also wraps `torch.distributed` (which in turn wraps a number of distribution technologies, including MPI) to permit the work to be distributed.

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

3.  Replace the `loss.backward()` call with `accelerator.backward(loss)`.

4.  Remove all the calls `.to(device)`: moving data to the appropriate device is handled automatically.

5.  Run the code with `accelerate launch` instead of `python`:

    ``` bash
    accelerate launch -m simple_cnn train --epochs 2
    ```

The `accelerate launch` command is technically optional, but it can take additional command-line arguments that give you control over the way the code is run. For example, to run the code on the CPU (even if there is a GPU), you can pass `--cpu` as a command-line argument:

``` bash
accelerate launch --cpu -m simple_cnn train --epochs 2
```

::: challenge

### Convert the code to use Accelerate

Change the code to make use of Accelerate and then run it on your machine.

Add, commit and push the changes to your copy of the github repo.

::: solution

The updated training code should look like:

``` python
from accelerate import Accelerate
from torch.optim import Adadelta
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from .model import SimpleCNN

def train(epochs, gamma=0.7):
    accelerator = Accelerate()
    model = SimpleCNN()
    optimizer = Adadelta(model.parameters())

    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    dataset = MNIST(transform=transform, train=True)
    training_data, validation_data = random_split(dataset, [split, 1 - split])

    training_loader = DataLoader(training_data, batch_size=64)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    model, optimizer, training_loader, scheduler = accelerator.prepare(
        model, optimizer, training_loader, scheduler
    )

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = nll_loss(output, target)

            accelerate.backward(loss)
            optimizer.step()

        scheduler.step()

    return model
```

And run with:

``` bash
accelerate launch -m simple_cnn train --epochs 2
```

:::

:::

### Preparing for Iridis

The compute nodes in the Iridis clusters don't have access to the wider internet, the call to `MNIST`, which will try to download the dataset will fail.

So we need to refactor the code so we can download the data to a local directory while running in a login node, and then load that data into memory on the compute node.

We add a module with code to do this:
``` python
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

DEFAULT_DATA_DIR = Path('data')

def download_mnist_data(data_directory=DEFAULT_DATA_DIR, redirect):
    data_directory.mkdir(parents=True, exist_ok=True)
    MNIST(data_directory, download=True)


def load_mnist_data(data_directory=DEFAULT_DATA_DIR) -> MNIST:
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    return MNIST(data_directory, transform=transform, train=True, download=False)
```

We replace the loading code in the `train` function:
``` python
def train(epochs, gamma=0.7):
    ...
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    dataset = MNIST(transform=transform, train=True)
    ...
```
with a call to the `load_mnist_data`:
``` python
from .data import load_mnist_data

def train(epochs, gamma=0.7):
    ...
    dataset = load_mnist_data()
    ...
```

We also add a command to the CLI to download the data.

::: challenge

### Download the Data

Make the changes to the code on your computer, and validate that they work by downloading the data and then running the training again.

Once it is working add and commit the changes.

::: solution

XXX Solution here

:::

:::

::: spolier

### Uploading data to Iridis with `scp`

In these examples we've downloaded the data from the internet. But what if your data is on *your* computer?

In this case the `scp` command is likely what you want. It is usually installed as part of SSH, and uses SSH to securely transfer files between computers.

:::

### Checkpoints and Resuming Training

XXX Add check-pointing code to the example along with code to resume training. Talk about Accelerate's save utility.

::: challenge

### Add check-points

XXX Example

:::

### Running with SLURM

XXX Writing the batch file

### Running with Mutliple GPUs

XXX Additional commands for Accelerate; changes to batch file.

###


::: keypoints


:::
