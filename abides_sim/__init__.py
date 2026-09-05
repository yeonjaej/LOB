import sys

# The shared venv's pip-editable install for abides-core/abides-markets (pointing at
# /Users/jwayeonjae/Documents/Claude/Projects/abides-jpmc-public/) relies on a .pth file
# that's sometimes not picked up at interpreter startup -- observed to be flaky while
# another session was concurrently modifying site-packages in this same venv. Inserting
# the source directories directly makes imports robust regardless of that, while still
# using the exact same (shared, patched) source tree.
for _p in (
    "/Users/jwayeonjae/Documents/Claude/Projects/abides-jpmc-public/abides-core",
    "/Users/jwayeonjae/Documents/Claude/Projects/abides-jpmc-public/abides-markets",
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
del _p, sys
