import hydra
import train  # train.pyからのインポート
from omegaconf import DictConfig
import os


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """hydraのメイン関数.

    Args:
        cfg (DictConfig): config.yamlのDictConfig
    """
    # トレーニングモジュールの実行
    train.run_training(cfg)


if __name__ == "__main__":
    main()
