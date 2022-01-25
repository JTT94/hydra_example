
import hydra

from run import run


@hydra.main(config_path="config", config_name="main")
def main(cfg):

    return run(cfg)

if __name__ == "__main__":
    main()