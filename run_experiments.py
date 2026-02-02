from src.config import load_config
from src.data.pipeline import build_and_save_processed_dataset


def main() -> None:
    cfg = load_config()
    ds = build_and_save_processed_dataset(cfg, force_raw=True)
    print(f"Processed dataset saved to: {ds.processed_path}")
    print(ds.df.head())
    print(f"Shape: {ds.df.shape}")


if __name__ == "__main__":
    main()
