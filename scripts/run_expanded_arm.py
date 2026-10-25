"""Fork a common checkpoint, train a longer arm, and score fixed final checkpoints."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    for key in ('student','teacher','checkpoint','data','config','output','destination'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    from scripts.worker import publish,write
    args.output.mkdir(parents=True,exist_ok=False)
    train=args.output/'train'
    common=['--student',str(args.student),'--data',str(args.data)]
    def call(label,argv):
        start=time.time()
        with (args.output/(label+'.log')).open('w') as stream:
            result=subprocess.run([sys.executable,*argv],stdout=stream,stderr=subprocess.STDOUT)
        write(args.output/(label+'-execution.json'),{'returncode':result.returncode,
            'start_unix_seconds':start,'end_unix_seconds':time.time()})
        publish(args.output,args.destination)
        if result.returncode:
            raise RuntimeError('Expanded arm failed at '+label)
    call('train',['-m','servedistill.train',*common,'--teacher',str(args.teacher),
        '--fork-parent',str(args.checkpoint),'--observe-host','--config',str(args.config),'--output',str(train)])
    state=json.loads((train/'checkpoint-0240/state.json').read_text())
    if state['step']!=240:raise RuntimeError('Final training step missing')
    call('final',['-m','servedistill.evaluate',*common,'--adapter',str(train/'checkpoint-0240'),
        '--phase','final','--output',str(args.output/'final')])
    selected=json.loads((train/'selected.json').read_text())
    write(args.output/'dev-selected.json',selected)
    if selected['path']!='checkpoint-0240':
        call('dev-selected-final',['-m','servedistill.evaluate',*common,
            '--adapter',str(train/selected['path']),'--phase','final',
            '--output',str(args.output/'selected-final')])
    write(args.output/'completed.json',{'completed':True,'primary_checkpoint':'checkpoint-0240',
        'test_used_for_selection':False,'dev_selected_secondary':selected['path']})
    publish(args.output,args.destination)


if __name__=='__main__':main()
