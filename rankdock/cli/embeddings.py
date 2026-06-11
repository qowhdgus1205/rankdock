"""Console entry point for embedding generation."""

import runpy


def main() -> None:
    try:
        runpy.run_module("rankdock.data.embeddings", run_name="__main__")
    except ImportError:
        runpy.run_module("data.embeddings", run_name="__main__")


if __name__ == "__main__":
    main()
