"""Re-export shim: the fixture generator lives in app.fortyguard.fixtures (these
are synthetic FortyGuard-shaped grids), but `python -m app.grid.fixtures --demo`
is the documented entrypoint, so keep it working from here too."""

from app.fortyguard.fixtures import main

if __name__ == "__main__":
    main()
