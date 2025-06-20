#!/usr/bin/env python3

# NOTE: This wrapper should not be necessary, but for some reason
# specifying a main script that is inside another package seems
# to confuse py2app in an unpredictable way, internally corrupting
# the zeroconf package.
from mavis.app import mavis_corrector

if __name__ == "__main__":
    mavis_corrector.main()
