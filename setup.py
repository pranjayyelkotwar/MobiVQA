#!/usr/bin/env python
"""
This setup.py script configures the installation of the VQA package.

It loads dependencies from several requirement files:
- "deps.txt" contains common dependencies.
- A system-specific requirement file ("req-tx2.txt" for tegra systems or "req-pc.txt" for PCs) provides extra dependencies.

The script prints out key steps to explain what is happening during setup.
"""

import platform
from pathlib import Path

from setuptools import find_packages, setup


def _load_requirements(req_file, comment_char="#"):
    print(f"Loading requirements from: {req_file}")
    with open(req_file, "r") as f:
        reqs = {
            line.strip()
            for line in f.readlines()
            if line.strip() and line.strip()[0] != comment_char
        }
    print(f"Found {len(reqs)} requirements in {req_file.name}.")
    return reqs


# Determine the root path of the project
PATH_ROOT = Path(__file__).parent
print(f"Project root determined as: {PATH_ROOT}")

# Detect system type to choose correct requirements file
system_type = platform.release()
print(f"Detected system platform release: {system_type}")

req_folder = PATH_ROOT / "requirements"
deps_file = req_folder / "deps.txt"
print(f"Common dependencies file set to: {deps_file}")

# Load common dependencies
deps_reqs = _load_requirements(deps_file)

# Decide system specific suffix: "tx2" for tegra systems and "pc" otherwise
sys_suffix = "tx2" if system_type.endswith("tegra") else "pc"
print(f"Using system-specific suffix: {sys_suffix}")

sys_req_file = req_folder / f"req-{sys_suffix}.txt"
print(f"System-specific dependencies file set to: {sys_req_file}")

# Load system-specific dependencies
extra_reqs = _load_requirements(sys_req_file)

# Combine all unique requirements from both files
all_reqs = deps_reqs.union(extra_reqs)
print(f"Total unique dependencies collected: {len(all_reqs)}")

# Setup configuration for packaging
setup(
    name="vqa",
    version="0.1.0",
    python_requires=">=3.6",
    install_requires=list(all_reqs),
    packages=find_packages(exclude=("data", "notes")),
)

print("Setup configuration complete. Proceeding with installation.")
