"""Install the released wheel in the community integration test environment."""

import sys
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory

with TemporaryDirectory() as folder:
    run(
        [
            "oras",
            "pull",
            "registry-1.docker.io/aperturedata/aperturedb-python:latest",
            "--output",
            folder,
        ],
        check=True,
    )
    (wheel,) = Path(folder).glob("aperturedb-*-py3-none-any.whl")
    pip = [sys.executable, "-m", "pip"]
    # A cached Poetry environment can already contain this version from PyPI.
    run([*pip, "uninstall", "--yes", "aperturedb"], check=True)
    run([*pip, "install", str(wheel)], check=True)
