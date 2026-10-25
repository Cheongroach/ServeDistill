"""Run fixed-bank timing and a short post-warmup Nsight capture on one GPU."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    for key in ('student','teacher','checkpoint','data','output','destination'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--worker-index',type=int,required=True)
    args=parser.parse_args()
    from scripts.worker import publish,write
    from scripts.profile_worker import install_nsys,locate
    args.output.mkdir(parents=True,exist_ok=True)
    common=['--student',str(args.student),'--teacher',str(args.teacher),
        '--checkpoint',str(args.checkpoint),'--data',str(args.data),'--output',str(args.output),
        '--worker-index',str(args.worker_index),'--banks','16','--repetitions','3']
    started=time.time()
    with (args.output/'benchmark.log').open('w') as stream:
        result=subprocess.run([sys.executable,'scripts/benchmark_microbatch.py',*common],
            stdout=stream,stderr=subprocess.STDOUT)
    write(args.output/'benchmark-execution.json',{'returncode':result.returncode,
        'start_unix_seconds':started,'end_unix_seconds':time.time()})
    publish(args.output,args.destination)
    if result.returncode:raise RuntimeError('Fixed-bank benchmark failed')
    nsys=locate('nsys')
    audit={'available':bool(nsys)}
    try:
        tools=args.output.parent.parent/f'nsight-tools-{args.worker_index}'
        tools.mkdir(parents=True,exist_ok=True)
        if not nsys:nsys,audit['install']=install_nsys(args.output.parent.parent/f'nsight-tools-{args.worker_index}')
        for microbatch in (1,4):
            capture=args.output/f'trace-m{microbatch}'
            with (args.output/f'nsys-m{microbatch}.log').open('w') as stream:
                result=subprocess.run([nsys,'profile','--trace=cuda,nvtx,osrt',
                    '--capture-range=cudaProfilerApi','--capture-range-end=stop','--sample=none',
                    '--cpuctxsw=none','--force-overwrite=true','--output='+str(capture),
                    sys.executable,'scripts/benchmark_microbatch.py',*common,
                    '--trace-bank','0','--trace-microbatch',str(microbatch)],
                    stdout=stream,stderr=subprocess.STDOUT,timeout=600)
            audit[f'm{microbatch}_returncode']=result.returncode
            if result.returncode == 0:
                for report in ('cuda_gpu_kern_sum','cuda_api_sum','nvtx_sum'):
                    with (args.output/f'm{microbatch}-{report}.csv').open('w') as stream:
                        subprocess.run([nsys,'stats','--report='+report,'--format=csv',
                            str(capture)+'.nsys-rep'],stdout=stream,stderr=subprocess.STDOUT,timeout=90)
    except (OSError,RuntimeError,subprocess.SubprocessError) as error:
        audit['error']=str(error)
    write(args.output/'trace-audit.json',audit)
    write(args.output/'worker-completed.json',{'completed':True,'benchmark_passed':True,
        'trace_status':audit,'boundary':'Trace failures do not overwrite benchmark evidence'})
    publish(args.output,args.destination)


if __name__=='__main__':main()
