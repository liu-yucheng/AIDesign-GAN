"""Package information.

Information items initially set to unknown and will update at runtime.
"""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

from importlib import metadata

# Aliases

_metadata = metadata.metadata

# -

pack_name = "aidesign-gan"
"""Package name."""
ver = "<unknown version>"
"""Version."""
author = "<unknown author>"
"""Author."""
cr = "<unknown copyright>"
"""Copyright."""
desc = "<unknown description>"
"""Description."""

_ver_key = "Version"
_author_key = "Author"
_cr_key = "License"
_desc_key = "Summary"

try:
    _pack_data = _metadata(pack_name)
    _pack_data = dict(_pack_data)

    ver = _pack_data[_ver_key]
    author = _pack_data[_author_key]
    cr = _pack_data[_cr_key]
    desc = _pack_data[_desc_key]
except Exception as _:
    pass
# end try
