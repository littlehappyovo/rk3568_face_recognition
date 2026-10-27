"""Opt-in packaging smoke test; run against a configured candidate build directory.

Creates and cleans a unique temporary static asset in this source checkout. Do not
run in a read-only review checkout or against a production service's build tree.
"""
from pathlib import Path
import argparse
import subprocess
import uuid


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build_directory',type=Path)
    parser.add_argument('--target-directory',type=Path,help='Executable directory for a multi-config generator')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    name='.resource-sync-'+uuid.uuid4().hex+'.txt'
    source=root/'server/face_access/static'/name
    target=(args.target_directory or args.build_directory)/'server/face_access/static'/name
    try:
        # The second build changes only server content. It must not require relinking.
        for content in ('asset-first-version\n','asset-second-version\n'):
            source.write_text(content)
            subprocess.run(['cmake','--build',str(args.build_directory),'--target','face_recognition_system','--parallel','2'],check=True)
            if not target.is_file() or target.read_text()!=content:
                raise RuntimeError('Incremental application build did not package current server assets')
        print('Server-only incremental asset packaging passed')
    finally:
        source.unlink(missing_ok=True);target.unlink(missing_ok=True)

if __name__=='__main__':main()
