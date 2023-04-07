"""
An example submitit script for running multiple jobs in parallel on a SLURM cluster
"""
import datetime
import shlex
import time
from pathlib import Path

import submitit
from tap import Tap


class SubmitItArgs(Tap):
    partition: str = 'your-slurm-partition'


# Args
args = SubmitItArgs().parse_args()

# Commands
commands = []
for data_path in Path('examples/natural-images').iterdir():
    name = data_path.name
    image_path = data_path / 'rgba.png'
    embeds_path = data_path / 'learned_embeds.bin'
    assert image_path.is_file() and embeds_path.is_file(), str(data_path)
    command = f"""python main.py --O --image_path {str(image_path)} --learned_embeds_path {str(embeds_path)} --run_name check-{name} --debug"""
    commands.append(command)

# Create executor
Path("slurm_logs").mkdir(exist_ok=True)
executor = submitit.AutoExecutor(folder="slurm_logs")
executor.update_parameters(
    tasks_per_node=1,
    timeout_min=90,
    slurm_partition=args.partition,
    slurm_gres="gpu:1",
    slurm_constraint="volta32gb",
    slurm_job_name="submititjob",
    cpus_per_task=8,
    mem_gb=32.0,
)

# Submitit via SLURM array
print('Start')
print(datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
start_time = time.time()
jobs = []
with executor.batch():
    print('SUBMITTING:')
    for command in commands:
        print(command)
        function = submitit.helpers.CommandFunction(shlex.split(command))
        job = executor.submit(function)
        jobs.append(job)

# Then wait until all jobs are completed:
outputs = [job.result() for job in jobs]
print(f'Finished all ({len(outputs)}) jobs in {time.time() - start_time:.1f} seconds')
print(datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
# Example: Finished all (14) jobs in 1530.5 seconds
