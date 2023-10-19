import sys

if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack
