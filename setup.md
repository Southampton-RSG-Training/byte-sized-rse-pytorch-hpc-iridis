---
title: Setup
---

::: prereq

- Shell with Git version control tool installed and the ability to navigate filesystem and run commands from within a shell
- Python version 3.13 or above installed
- An Iridis account
- An understanding of Python syntax to be able to read code examples

:::

## Software Setup

### Bash and Git

On macOS and Linux, some version of a shell (e.g. `bash`) with Git will be available by default and no installation is needed.

If you do not have a bash shell installed on your system and require assistance with the installation, you can take a look at [the instructions provided by Software Carpentry](https://swcarpentry.github.io/python-novice-inflammation/#install-python)
for installing [shell](https://carpentries.github.io/workshop-template/install_instructions/#the-bash-shell) and [Git](https://carpentries.github.io/workshop-template/install_instructions/#git-1).

### Python

Python version 3.13 or above is required. Type `python -V` (or `python3 -V` on Mac) at your shell prompt and press enter to see what version of Python is installed on your system.
If you do not have Python installed on your system and require assistance with the installation, you can take a look at [the instructions provided by Software Carpentry](https://swcarpentry.github.io/python-novice-inflammation/#install-python)
for installing Python in preparation for undertaking their Python lesson.

### Access to Iridis

You will need access to Iridis for this session.

If you do not have access to Iridis, the process of getting access to the system is straightforward for University of Southampton Staff and Students. Use of the system is *free* at the point of use, and there is a [short application form](https://sotonac.sharepoint.com/teams/HPCCommunityWiki/SitePages/Connecting-to-Iridis.aspx) to be filled in.

You will also need to ensure that you have an SSH client installed on your system, and have set up your SSH keys for Iridis as described in the above wiki page.

### VPN Access if Remote

If you are attending the session while not connected to the University of Southampton, you will need to be connected to the University of Southampton VPN.

### Example Code

The course involves an [activity which can be found on GitHub](https://github.com/Southampton-RSG-Training/byte-sized-rse-python-hpc-example).

You can save time during the course by downloading this material beforehand on your local machine as well as in your account on the Iridis 6 login node:

``` bash
git clone https://github.com/Southampton-RSG-Training/byte-sized-rse-python-hpc-example.git
```
