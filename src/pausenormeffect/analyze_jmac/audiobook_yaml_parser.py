import yaml


def extract_yaml_values(yaml_data):
    """Extracts 'character' and 'sent' and 'time' and 'to whom' values from nested dictionaries.

    Parameters:
        yaml_data (dict): A nested dictionary containing the data.

    Returns:
        list: A list of dictionaries containing 'character' and 'sent' and 'time' and 'to whom'.
    """
    extracted_data = []

    # Loop through each chapter
    for chapter_key, chapter_value in yaml_data.items():
        # Loop through each paragraph in the chapter
        for paragraph_key, paragraph_value in chapter_value.items():
            # Loop through each style in the paragraph
            for style_key, style_value in paragraph_value.items():
                # Loop through each entry in the style list
                for entry in style_value:
                    # Extract 'character' and 'sent' and 'time' and 'to whom'
                    character = entry.get("character", "N/A")
                    sent = entry.get("sent", "N/A")
                    time = entry.get("time", "N/A")
                    to_whom = entry.get("to whom", "N/A")
                    # Append the extracted data to the list
                    extracted_data.append(
                        {
                            "character": character,
                            "sent": sent,
                            "time": time,
                            "to whom": to_whom,
                        }
                    )

    return extracted_data


def read_yaml(yaml_path):
    """Reads a YAML file and returns the data as a dictionary.

    Parameters:
        yaml_path (str): The path to the YAML file.

    Returns:
        dict: The YAML data.
    """
    with open(yaml_path) as file:
        yaml_data = yaml.safe_load(file)

    return yaml_data


def extract_yaml_data(yaml_path):
    """Extracts 'character' and 'sent' and 'time' and 'to whom' values from a YAML file.

    Parameters:
        yaml_path (str): The path to the YAML file.

    Returns:
        list: A list of dictionaries containing 'character' and 'sent' and 'time' and 'to whom'.
    """
    yaml_data = read_yaml(yaml_path)
    extracted_data = extract_yaml_values(yaml_data)

    return extracted_data


if __name__ == "__main__":
    yaml_path = "data/audiobook/audiobook.yaml"
    yaml_path = (
        "/data/corpus/J-MAC/txt-fixed/楠山正雄/おおかみと七ひきのこやぎ/おおかみと七ひきのこやぎ (1)/all.yaml"
    )
    extracted_data = extract_yaml_data(yaml_path)
    print(extracted_data)
