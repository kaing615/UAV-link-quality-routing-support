"""Preserve a command's direct child exit code and unfiltered output, exclusively."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER);args=p.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:raise ValueError('Command required')
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.time()
    with (args.output/'stdout.log').open('x',encoding='utf-8') as stdout,(args.output/'stderr.log').open('x',encoding='utf-8') as stderr:
        result=subprocess.run(command,stdout=stdout,stderr=stderr,check=False)
    (args.output/'command.json').write_text(json.dumps(dict(command=command,returncode=result.returncode,start_unix_s=start,end_unix_s=time.time()),indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),returncode=result.returncode)))
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
