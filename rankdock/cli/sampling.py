"""Console entry point for initial sampling."""

import runpy


def main() -> None:
    try:
        runpy.run_module("rankdock.sampling", run_name="__main__")
    except ImportError:
        runpy.run_module("sampling", run_name="__main__")


if __name__ == "__main__":
    main()
