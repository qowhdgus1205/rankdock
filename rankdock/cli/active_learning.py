"""Console entry point for the RankDock active-learning loop."""

try:
    from rankdock.active_learning import parse_args, run_bo
except ModuleNotFoundError:
    from active_learning import parse_args, run_bo


def main() -> None:
    run_bo(parse_args())


if __name__ == "__main__":
    main()
