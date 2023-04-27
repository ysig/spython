# spython : Making sbatch more user-friendly for python users.
Do you still want to write `sbatch` files yourself?
`spython` is meant to improve the experience of python users of Jean-Zay and is an alpha-development stage (optimal for a single user and now open to many).

## Motivation
Jean Zay is one of the coolest thing that has happened to France in terms of computing (optimized, well managed, abundant, open). Its support team is the best example of a French institution: extremely polite, trying to help you in everything and quick to answer (unfortunately not how French beaurocracy looks on average). 

However, there are two main problems with using Jean-Zay: it's like coding in C instead of python and it's counter-productive. Most of this comes from the fact of both complex rules that need to be read and understood (similar to law - but more pragmatic) but also for most operations it's simply a matter of copy pasting sbatch scripts around that are hard to read, understanding and digesting usage regulations and learning how to allocate modules, priorities and compute nodes.

Here is where this script comes in. `spython` is a simple script that does all that for you and lifts interaction with `sbatch` instead to a more higher level of commands, implementing the Jean-Zay [document](http://www.idris.fr/media/eng/ia/guide_nouvel_utilisateur_ia-eng.pdf). For example, you can directy write the number of GPUs you want ot use the memory of each in gb's or the total cpu ram with any of these arguments `--ngpu 2` `--gb 16` `--ram l` (if arguments are missing it can self configure itself to optimal values) or e.g. select time `--t 19` or place it easily to `--debug` priority quickly. It has even a function called `--live` where the user experience of running a program imitates that of directly running python!* Output logs and scripts are automatically saved under a log directory in `$STORE` and tagged according to a tag `--tag` hyperparameter. 

\*without ctrl-c for practical purposes.

### Contribute
We hope that the Jean-Zay team & community get's to converge in high-level scripts or libraries that help improve quick and wasteless integration. Using such scripts high level practices can directly get implemented on a program level (instead of documentation) so that Jean-Zay is used much more optimally.

As this package sprawls from the user-experience of a certain user it can be limited to their explored use-cases. However you can easily incorporate your use-case to this package, by making pull-requests. If this package becomes useful, documentation and a python installation will be the next steps.

## Usage
To use `spython` you firstly need to install it in your system, which takes not more than 10 seconds. It carries no-python dependencies and can work on native jean-zay python.

### Installation
Currently the project requires manual installation, which can be done in 2 simple steps:

```python
wget -O ~/jean-zay.py <url> # installs the main-script
wget -O - <url> >> ~/.bashrc # installs macros
```

After reopening a terminal you will be exposed to the cli-commands: 
- `spython`
- `saccelerate`
- (hopefully more to be contributed in the future).

### Examples
For all our examples we will suppose that what we really want is to run a `train.py`.

#### Basic example
imply append to your python script command `:` adding a set of Jean Zay allocation specific arguments after it:

```bash
spython train.py <ARGS> : --ngpu 2 --ncpu 10 --module-load pytorch-gpu/py3/1.11.0 --tag EXP --gb 32 --env geography --post-script jean-zay/server.sh --email <your-email> --name test
```

From this script we can firstly notice that we we allocate 2 gpus and 10 cpus per task.
We then `module load` a certain package (but multiple can be chained) and `conda activate geography` after the module is loaded.

We also copy paste server a script `jean-zay/server.sh` to run after environment is loaded and send emails to a specific address about the status of the experiment using `test` as a name.

Running directly:
`spython --ngpu 2 --ncpu 10 --module-load pytorch-gpu/py3/1.11.0 --tag EXP --gb 32 --post-script jean-zay/server.sh`

will just print a command to run to generate the same environment:

`srun --pty --job-name=test --constraint=v100-32g --nodes=1 --gres=gpu:2 --cpus-per-task=10 --ntasks=2 --ntasks-per-node=2 --time=72:00:00 --qos=qos_gpu-t4 -A hkt@v100 bash`

but omitting any in environment commands like running the `server.sh` or module-load.

#### Help
For more covered functionality below follows the detailed `--help` output of the script:

```bash
usage: jean-zay.py [-h] [--gb {16,32,40,80}] [--ram {l,m,h}] [--debug] [--ngpu NGPU] [--ncpu NCPU] [--time TIME] [--name NAME] [--module-load MODULE_LOAD [MODULE_LOAD ...]] [--ntasks NTASKS]
                   [--ntasks-per-node NTASKS_PER_NODE] [--submission-dir SUBMISSION_DIR] [--error-file ERROR_FILE] [--output-file OUTPUT_FILE] [--script-file SCRIPT_FILE] [--conda_path CONDA_PATH]
                   [--command COMMAND] [--email EMAIL] [--env ENV] [--preload] [--prepost] [--account ACCOUNT] [--tag TAG] [--path PATH] [--live] [--post-script POST_SCRIPT]

optional arguments:
  -h, --help            show this help message and exit
  --gb {16,32,40,80}    Number of GBs GPU to reserve (default: None)
  --ram {l,m,h}         Memory Mode (default: None)
  --debug               Debug (default: False)
  --ngpu NGPU, -g NGPU  Num GPUs (default: 1)
  --ncpu NCPU, -c NCPU  Num CPUs (default: 10)
  --time TIME, -t TIME  Max Time (hrs) (default: 72)
  --name NAME, -n NAME  Job Name (default: $USER)
  --module-load MODULE_LOAD [MODULE_LOAD ...], -ml MODULE_LOAD [MODULE_LOAD ...]
  --ntasks NTASKS       Number of MP tasks (default: None)
  --ntasks-per-node NTASKS_PER_NODE
                        Number of tasks per node (default: None)
  --submission-dir SUBMISSION_DIR
                        Directory where submission files and outputs are stored (default: None)
  --error-file ERROR_FILE, -e ERROR_FILE
                        Error file (default: log.txt)
  --output-file OUTPUT_FILE, -o OUTPUT_FILE
                        Output file (default: log.txt)
  --script-file SCRIPT_FILE, -s SCRIPT_FILE
                        Script file (default: script.txt)
  --conda_path CONDA_PATH
                        Path of conda (default: None)
  --command COMMAND     Main command to execute (default: python)
  --email EMAIL         Email of user (default: None)
  --env ENV             Environment (default: None)
  --preload             Preload - if not set modules wil be purged (default: True)
  --prepost             Set on a prepost node (default: False)
  --account ACCOUNT     Manually set account name (default: None)
  --tag TAG             Set an experiment tag (default: None)
  --path PATH           Set explicit path from which to start the experiment (default: os.getcwd())
  --live                Debug (default: False)
  --post-script POST_SCRIPT
                        A script to be executed before the main command (default: None)
```

#### Behind the scenes.
As a BTS example we will show what happens with our current support `saccelerate`.
The user will simply run:

`saccelerate <ARGS> : --ngpu 2 --ncpu 10 --module-load pytorch-gpu/py3/1.11.0 --tag <TAG> --gb 32 --post-script jean-zay/server.sh --debug --live`

but the following script gets generated:

For example 
On `/gpfsstore/rech/hkt/$USER/submissions/EXP/20230427-153034/` you will find two files: 
- `log.txt` on the same address will save all std output.
- `script.txt` contains the `sbatch script`
	```
	#!/bin/bash
  #SBATCH --hint=nomultithread
  #SBATCH --distribution=block:block
  #SBATCH --job-name=$USER
  #SBATCH --constraint=v100-32g
  #SBATCH --nodes=1
  #SBATCH --gres=gpu:2
  #SBATCH --cpus-per-task=10
  #SBATCH --ntasks=1
  #SBATCH --ntasks-per-node=1
  #SBATCH --time=1:00:00
  #SBATCH --qos=qos_gpu-dev
  #SBATCH --output=/gpfsstore/rech/hkt/$USER/submissions/<TAG>/20230425- 222102/log.txt
  #SBATCH --error=/gpfsstore/rech/hkt/$USER/submissions/<TAG>/20230425-222102/log.txt
  #SBATCH -A hkt@v100
  
  source /linkhome/rech/genlgm01/$USER/.bashrc
  module purge
  module load pytorch-gpu/py3/1.11.0
  set -x

  cd /gpfsdswork/projects/rech/hkt/<user-id>/<PROJECT-PATH>

  module load singularity

  singularity exec --bind $SCRATCH:/data $SINGULARITY_ALLOWED_DIR/tileserver.sif node /usr/src/app/ /data/planet.mbtiles &> /dev/null &

  export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
  export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  export MASTER_PORT=6000

  srun accelerate launch --num_processes 2 --multi_gpu --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT train.py 
  ```

Do you still want to write `sbatch` files yourself?