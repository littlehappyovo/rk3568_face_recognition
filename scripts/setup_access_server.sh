#!/usr/bin/env bash
# Install access-service dependencies in a project-only virtual environment.
set -euo pipefail
access_project_dir="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
access_venv_dir="${FACEAPP_ACCESS_VENV:-$access_project_dir/server/.venv}"
access_python="${FACEAPP_ACCESS_BOOTSTRAP_PYTHON:-python3}"
env -u PYTHONPATH -u PYTHONHOME "$access_python" -I -c 'import sys; assert sys.version_info >= (3, 10), "Python3.10 or newer required"'
env -u PYTHONPATH -u PYTHONHOME "$access_python" -I -m venv "$access_venv_dir"
"$access_venv_dir/bin/python" -I -m pip --require-virtualenv install -r "$access_project_dir/server/requirements.lock"
"$access_venv_dir/bin/python" -I -c 'import flask, waitress, PIL, sqlite3; print("管理后台依赖已就绪")'
printf 'FACEAPP_ACCESS_PYTHON=%s\n' "$access_venv_dir/bin/python"
