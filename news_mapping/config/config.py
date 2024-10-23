import yaml


def read_yaml_file(file_path):
    """
    Reads a YAML file and returns the contents as a Python dictionary.

    :param file_path: Path to the YAML file
    :return: Contents of the YAML file as a dictionary
    """
    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None
    except yaml.YAMLError as e:
        print(f"Error reading YAML file: {e}")
        return None
