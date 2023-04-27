import time as time
import os, sys
import warnings

from math import ceil
from os import getcwd, sep, makedirs
from os.path import join, dirname, expandvars, expanduser, split, isdir, abspath, basename
from subprocess import call

TABLE = """
Partition QoS Time limits  Resource limits
           per job per user per project par QoS
CPU
qos_cpu-t3* 20 h 20480 cores 48000 cores 48000 cores
qos_cpu-t4 100 h 160 cores 1280 cores 1280 cores 5120 cores
qos_cpu-dev 2 h 5120 cores 5120 cores 5120 cores 48000 cores
GPU
qos_gpu-t3* 20 h 512 GPUs 1024 GPUs 1024 GPUs
qos_gpu-t4 100 h 16 GPUs 128 GPUs 128 GPUs 512 GPUs
qos_gpu-dev 2 h 32 GPUs 32 GPUs 32 GPUs 512 GPUs

Quadri-GPU V100 + RAM GPU 16 GB --constraint v100-16g
Quadri-GPU V100 + RAM GPU 32 GB --constraint v100-32g
Octo-GPU V100 --partition=gpu_p2
Octo-GPU V100 + RAM CPU 384 GB --partition=gpu_p2s
Octo-GPU V100 + RAM CPU 768 GB --partition=gpu_p2l
Octo-GPU A100 --partition=gpu_p4
"""

class JeanZay(object):
    __ORDER__ = ['command', 'debug', 'name', 'tag', 'submission_dir', 'prepost', 'gb', 'ram', 'ngpu', 'ncpu', 'ntasks', 'ntasks_per_node', 'time', 'qos', 'output_file', 'error_file', 'script_file', 'conda_path', 'env', 'preload', 'module_load', 'post_script', 'email', 'account', 'live', 'path']
    __STATIC__ = [
        '#SBATCH --hint=nomultithread',
        '#SBATCH --distribution=block:block',
    ]

    def __init__(self, args):
        self.script = ["#!/bin/bash", ""]
        self.args = []
        for k in self.__ORDER__:
            if k != 'qos':
                getattr(self, k)(args[k])
            else:
                getattr(self, k)()
        self.script += self.__STATIC__

    def command(self, val):
        self.command_ = val

    def gb(self, val):
        self.gb_ = val

    def prepost(self, val):
        self.prepost_ = bool(val)
        if self.prepost_:
            self.args.append('--partition=prepost')

    def tag(self, val):
        self.tag_ = val

    def ram(self, val):
        if self.prepost_:
            return

        partition, constraint = '', ''

        self.octo, self.a100_ = False, False
        if self.gb_ == 16:
            constraint = 'v100-16g'
            self.ncpus_max = 40
        elif self.gb_ == 32 and val is None:
            constraint = 'v100-32g'
            self.ncpus_max = 40
        elif self.gb_ == 40:
            self.a100_ = True
            self.p5 = False
        elif self.gb_ == 80:
            self.p5 = True
            self.a100_ = True
            if val is None:
                val = 'm'

        if val == 'm':
            if self.gb_ is None:
                self.a100_ = True
            else:
                assert self.a100_, 'For ram \'m\' setting only gb should be set to 40 (corresponding to A100)'

        if self.a100_:
            if val == 'l':
                warnings.warn('Ignoring low memory as in A100 mode')

            if val == 'm':
                assert getattr(self, 'p5', True)
                self.p5 = True
                partition = 'gpu_p5'
            else:
                assert getattr(self, 'p5', False) == False
                partition = 'gpu_p4'
            
            self.octo = True
            self.ncpus_max = 48
        elif val == 'l':
            partition = 'gpu_p2s'
            self.octo = True
            self.ncpus_max = 24
        elif val == 'h' and not self.a100_:
            partition = 'gpu_p2l'
            self.octo = True
            self.ncpus_max = 24

        if len(partition):
            self.args.append(f"--partition={partition}")

        if len(constraint):
            assert not len(partition)
            self.args.append(f"--constraint={constraint}")

        if getattr(self, 'p5', False):
            self.args.append("-C a100")


    def name(self, val):
        self.args.append(f"--job-name={val}")

    def ngpu(self, val):
        self.nodes = 1
        if self.prepost_:
            self.ngpu_ = 1
            self.ng_ = 1
            return

        self.ngpu_ = val
        if self.ngpu_ > 0:
            node_size = (8 if self.octo else 4)
            self.nodes = ceil(self.ngpu_ / float(node_size))
            ng = min(ceil(self.ngpu_ / self.nodes*1.0), node_size)
            self.ng_ = ng
            self.args.append(f"--nodes={self.nodes}")
            self.args.append(f"--gres=gpu:{ng}")

    def debug(self, val):
        self.debug_ = val

    def ntasks(self, val):
        if val is None:
            if self.ngpu_ > 0 and self.command_ != 'accelerate':
                val = self.ngpu_
            else:
                val = 1
        self.args.append(f"--ntasks={val}")

    def ntasks_per_node(self, val):
        if val is None:
            if self.ngpu_ > 0 and self.command_ != 'accelerate':
                val = self.ng_
            else:
                val = 1
        self.args.append(f"--ntasks-per-node={val}")

    def ncpu(self, val):
        if hasattr(self, 'ncpus_max'):
            self.ncpus_max = (self.ncpus_max * self.nodes)//self.ngpu_
            if val == -1:
                val = self.ncpus_max
            else:
                if val > self.ncpus_max:
                    warning.warn(f'adjusting val={val} to {self.ncpus_max}')
                val = min(self.ncpus_max, val)
        else:
            assert val != -1, 'ncpus_max is not set we don\'t know the default partition'

        self.args.append(f"--cpus-per-task={val}")

    def time(self, val):
        self.time_ = int(val)

    def qos(self):
        cpu = ('gpu' if self.ngpu_ > 0 else 'cpu')
        if self.debug_:
            assert self.ngpu_ <= 32, 'During debug max gpus can be less than 32'
            t = 'dev'
            self.time_ = min(2, self.time_)
        elif self.time_ > 20:
            t = 't4'
        else:
            t = 't3'

        assert self.time_ <= 100, 'Impossible. For increasing the gpus consult table page 28 here: http://www.idris.fr/media/eng/ia/guide_nouvel_utilisateur_ia-eng.pdf'

        qos = f"{cpu}-{t}"
        self.args.append(f"--time={self.time_}:00:00")
        self.args.append(f"--qos=qos_{qos}")

    def output_file(self, val):
        val = join(self.submission_dir_, val)
        self._make_parent_dir(val)
        self.of = val
        self.args.append(f'--output={val}')

    def error_file(self, val):
        val = join(self.submission_dir_, val)
        self._make_parent_dir(val)
        self.ef = val
        self.args.append(f'--error={val}')

    def script_file(self, val):
        val = join(self.submission_dir_, val)
        self._make_parent_dir(val)
        self.script_file = val

    def conda_path(self, val):
        self.conda_path = (join(expandvars("$WORK"), 'miniconda3') if val is None else val)
        assert isdir(self.conda_path) or self.conda_env is None, 'Path and env should both exist.'

    def env(self, val):
        self.conda_env = val

    def preload(self, val):
        self.purge = not bool(val)

    def module_load(self, val):
        self.modules = val

    def account(self, val):
        if val is None:
            val = 'hkt@v100'

        if '@' in val:
            a, b = val.split('@')
            if self.ngpu_ == 0:
                val = '@'.join([a, 'cpu'])
            elif getattr(self, 'p5', False):
                val = '@'.join([a, 'a100'])
            else:
                val = '@'.join([a, 'v100'])

        self.args.append(f'-A {val}')

    def live(self, val):
        self.live_ = bool(val)

    def email(self, val):
        if val is not None:
            self.args.append(f'--mail-user={val}')
            self.args.append(f'--mail-type=ALL')

    def path(self, val):
        self.path_ = val

    def post_script(self, val):
        self.post_script_ = val

    def _make_parent_dir(self, fname):
        drn = dirname(fname)
        makedirs(drn, exist_ok=True)

    def submission_dir(self, val):
        if self.tag_ is None and val is not None:
            warnings.warn(f'tag will be overidden by {val}')

        if val is None:
            date = time.strftime("%Y%m%d-%H%M%S")
            val = join(expandvars('$STORE'), 'submissions')
            if self.tag_ is not None:
                val = join(val, self.tag_)
            val = join(val, date)

        self.submission_dir_ = val

    def __call__(self, command=None):
        if command is None:
            args = [a for a in self.args if all(c not in a for c in ['error', 'input', 'output'])]
            command = ' '.join(['srun', '--pty'] + args + ['bash'])
            print(command)
            #os.system(' '.join(['srun', '--pty'] + args + ['bash']))
            return

        self.script += [f'#SBATCH {arg}' for arg in self.args]
        self.script += [f'source ' + expanduser('~/.bashrc')]

        if not self.purge:
            self.script += ['module purge']

        for module in self.modules:
            self.script += [f'module load {module}']

        if self.conda_env is not None:
            self.script += ['conda activate ' + join(self.conda_path, 'envs', self.conda_env)]
 
        self.script += ['set -x']
        self.script += ['', f'cd ' + self.path_]
        if self.post_script_ is not None:
            self.script += [''] + list(open(self.post_script_, 'r').readlines())
        if self.command_ == 'accelerate':
            assert self.ngpu_ >= 1
            self.script += [
                '',
                'export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`',
                'export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)',
                'export MASTER_PORT=6000',
            ]
            multi_gpu = (' ' if self.ngpu_ <= 1 else ' --multi_gpu ')
            command_ = f'accelerate launch --num_processes {self.ngpu_}{multi_gpu}--main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT'
            if self.nodes > 1:
                raise ValueError('JZ is not yet configured for accelerate with more than 1 ports')
        elif self.command_ == 'python':
            command_ = 'python -u'
        else:
            command_ = self.command_
        self.script += ['', f'srun {command_} ' + ' '.join(command)]
     
        self._make_parent_dir(self.script_file)
        with open(self.script_file, 'w') as f:
            script = '\n'.join(self.script)
            print(script, file=f)
            print(script)

        call(['sbatch', self.script_file])
        if self.live_:
            print('tail -f -n 20 ' + self.of)
            call(['touch', self.of])
            call(['tail', '-f', '-n', '20', self.of])
        else:
            print('\n--------------------------\nto trace run:\n\n' + 'tail -f -n 20 ' + self.of)


def argparse(args):
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gb', help='Number of GBs GPU to reserve',  choices=[16, 32, 40, 80], type=int, default=None)
    parser.add_argument('--ram', help='Memory Mode',  choices=['l', 'm', 'h'], type=str, default=None)
    parser.add_argument('--debug', help='Debug', action='store_true')
    parser.add_argument('--ngpu', '-g', help='Num GPUs', type=int, default=1)
    parser.add_argument('--ncpu', '-c', help='Num CPUs', type=int, default=10)
    parser.add_argument('--time', '-t', help='Max Time (hrs)', type=int, default=72)
    parser.add_argument('--name', '-n', help='Job Name', type=str, default=basename(abspath(expanduser('~/'))))
    parser.add_argument('--module-load', '-ml', nargs='+', default=[])
    parser.add_argument('--ntasks', help='Number of MP tasks', default=None)
    parser.add_argument('--ntasks-per-node', help='Number of tasks per node', default=None)
    parser.add_argument('--submission-dir', help='Directory where submission files and outputs are stored', type=str, default=None)
    parser.add_argument('--error-file', '-e', help='Error file', type=str, default='log.txt')
    parser.add_argument('--output-file', '-o', help='Output file', type=str, default='log.txt')
    parser.add_argument('--script-file', '-s', help='Script file', type=str, default='script.txt')
    parser.add_argument('--conda_path', help='Path of conda', type=str, default=None)
    parser.add_argument('--command', help='Main command to execute', type=str, default='python')
    parser.add_argument('--email', help='Email of user', type=str)
    parser.add_argument('--env', help='Environment', type=str, default=None)
    parser.add_argument('--preload', help='Preload - if not set modules wil be purged', action='store_false')
    parser.add_argument('--prepost', help='Set on a prepost node', action='store_true')
    parser.add_argument('--account', help='Manually set account name', default=None)
    parser.add_argument('--tag', help='Set an experiment tag', default=None)
    parser.add_argument('--path', help='Set explicit path from which to start the experiment', default=getcwd())
    parser.add_argument('--live', help='Debug', action='store_true')
    parser.add_argument('--post-script', help='A script to be executed before the main command', default=None)

    return vars(parser.parse_args(args))


def args_split():
    for i in range(len(sys.argv)):
        if sys.argv[i] == ':':
            break
    return sys.argv[1:i], sys.argv[i+1:]

if __name__ == "__main__":
    if any(sys.argv[i] == ':' for i in range(len(sys.argv))):
        start, end = args_split()
        arguments = argparse(end)
        JeanZay(arguments)(start)
    else:
        arguments = argparse(sys.argv[1:])
        JeanZay(arguments)()
