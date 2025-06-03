import os
import yaml

from inference import inference


def main():

    with open('parameters_inference.yaml') as file:
        parameters_inf = yaml.load(file, yaml.FullLoader)

    model_dir = parameters_inf["model_dir"]
    dataset_dir = parameters_inf["dataset_path"]

    with open(os.path.join(model_dir, 'parameters.yaml')) as file:
        parameters = yaml.load(file, yaml.FullLoader)

    inference(parameters, dataset_dir)


if __name__ == "__main__":
    main()
