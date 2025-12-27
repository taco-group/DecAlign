from main import DMD_run

if __name__ == "__main__":
    DMD_run(
        model_name='decalign',
        dataset_name='mosi',
        mode='test',
        seeds=[1111],
        model_save_dir="./pt",
        res_save_dir="./result",
        log_dir="./log"
    )