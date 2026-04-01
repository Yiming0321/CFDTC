# to-do: 需要重新做

from train_mlp import inference

if __name__ == "__main__":

    best_model_path = r"./model_path.pth" # replace to your absolute path
    data_path = r"./data_path" # replace to your absolute path
    result_dir = r"./output" # replace to your absolute path

    prediction_results = inference(
        model_path = best_model_path,
        data = data_path,
        save_dir = result_dir
    )