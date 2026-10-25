"""Apply verified expansion code/data and copy the immutable common parent locally."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--code',type=Path,required=True)
    parser.add_argument('--checkpoint-source',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with tarfile.open(args.archive) as archive:
        for member in archive.getmembers():
            path=Path(member.name)
            if path.is_absolute() or '..' in path.parts or not(member.isdir() or member.isfile()):
                raise ValueError('Unsafe expansion payload')
        archive.extractall(args.code)
    manifest=json.loads((args.code/'expansion-files.json').read_text())
    for name,digest in manifest.items():
        if hashlib.file_digest((args.code/name).open('rb'),'sha256').hexdigest()!=digest:
            raise RuntimeError('Expansion source hash mismatch: '+name)
    shutil.copytree(args.checkpoint_source,args.checkpoint)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'expansion-source-hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'common-parent-state.json').write_text((args.checkpoint/'state.json').read_text())
    print('Expansion sources/data and common checkpoint copied and verified',flush=True)


if __name__=='__main__':main()
