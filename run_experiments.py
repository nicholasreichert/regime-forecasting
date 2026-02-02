from src.config import load_config


def main() -> None:
    cfg = load_config()
    print(
        f"Loaded config: ticker={cfg.data.ticker}, "
        f"horizons={cfg.targets.horizons}, "
        f"K_values={cfg.hmm.K_values}, seed={cfg.project.seed}"
    )

    # create dirs
    _ = cfg.resolve_dir(cfg.paths.raw_dir)
    _ = cfg.resolve_dir(cfg.paths.processed_dir)
    _ = cfg.resolve_dir(cfg.paths.reports_dir)
    _ = cfg.resolve_dir(cfg.paths.figures_dir)
    print("Directories ensured.")


if __name__ == "__main__":
    main()
