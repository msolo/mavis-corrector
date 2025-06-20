"""
Usage:
    python setup.py py2app
"""

import subprocess
import sys

# The code complexity of transformers requires that we increase
# the recusion limit for the dependency analysis.
sys.setrecursionlimit(5000)
from setuptools import setup

# Record the version we build on for compatibility.
os_release_version = subprocess.check_output(['/usr/bin/sw_vers', '-productVersion']).decode().strip()

# NOTE: This is very fragile. If the main script is in a package, it
# can fail to correctly include zeroconf - the package is internally
# mangled and it's not clear why.
APP = ["./MavisCorrector.py"]

OPTIONS = {
    "packages": [
        "mavis",
        "zeroconf",
    ],
    "excludes": [
        "setuptools",
    ],
    "resources": [
        "models",
    ],
    "extension": ".plugin",
    "plist": {
        "CFBundleIdentifier": "com.hiredgoons.MavisCorrector.plugin",
        "CFBundleVersion": "0.1.2",
        "Py2AppBuildOSRelease": os_release_version,
    },
    # This generates so much file copying that it makes spotlight
    # re-index like mad, so us a noindex directory.
    "dist_dir": "derived.noindex/dist",
    "bdist_base": "derived.noindex/build",
}

setup(
    app=APP,
    options={
        "py2app": OPTIONS,
    },
    setup_requires=["py2app"],
)
