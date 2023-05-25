from distributed.deploy.ssh import Worker, Scheduler 
from distributed.deploy.spec import SpecCluster
import socket
    

def get_config(hostname): 
    pinac_options = dict(
        nthreads = 1, 
        n_workers = 2, 
        memory_limit = '16 GiB',
        local_directory = '/cw/dtailocal/',
    )

    himec_options = dict(
        nthreads = 1, 
        n_workers = 7, 
        memory_limit = '16 GiB',
        local_directory = '/cw/dtailocal/',
    )
    modern_himec_options = dict(
        nthreads = 1, 
        n_workers = 7, 
        memory_limit = f"16 GiB",
        local_directory = '/cw/dtailocal/',
    )
    if 'pinac' in hostname: 
        return pinac_options
    if 'himec07' in hostname or 'himec08' in hostname: 
        return modern_himec_options 
    if 'himec' in hostname: 
        return himec_options 
    


def get_dask_cluster(pinac_numbers, himec_numbers): 
    # process worker arguments
    host_names = [f'pinac{o}.cs.kuleuven.be' for o in pinac_numbers] + [f'himec{o:02d}.cs.kuleuven.be' for o in himec_numbers]
    configs = [get_config(host) for host in host_names]
    ips = [socket.gethostbyname(name) for name in host_names]
    
    # constants
    connect_options=dict(username= 'jonass', known_hosts=None)
    scheduler_options: dict = {"port": 0, "dashboard_address": ":8787"}
    worker_class: str = "distributed.Nanny"
    remote_python = "/cw/dtaijupiter/NoCsBack/dtai/jonass/miniconda/envs/energyville/bin/python"


    scheduler = {
            "cls": Scheduler,
            "options": {
                "address": 'localhost',
                "connect_options": connect_options,
                "kwargs": scheduler_options,
                "remote_python": remote_python,
            }, 
        }

    workers = {
            i: {
                "cls": Worker,
                "options": {
                    "address": host,
                    "connect_options": connect_options,
                    "kwargs": config,
                    "worker_class": worker_class,
                    "remote_python": remote_python,
                },
            }
            for i, (config, host) in enumerate(zip(configs, ips))
        }
    return SpecCluster(workers, scheduler, name="SSHCluster")

