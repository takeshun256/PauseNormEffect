from typing import Dict, List, Union

import yaml


class AudiobookScriptExtractor:
    def __init__(self, yaml_filepath: str):
        self.yaml_filepath = yaml_filepath
        self.yaml_content = self._load_yaml_file()

    def _load_yaml_file(self) -> Dict[str, Union[Dict, List[Dict]]]:
        """Load YAML file content."""
        with open(self.yaml_filepath, "r") as file:
            return yaml.safe_load(file)

    def extract_reading_scripts(self) -> List[Dict[str, str]]:
        """Extract reading script details like character, sentence, time, and recipient."""
        extracted_scripts = []
        for _, chapter_content in self.yaml_content.items():
            for _, paragraph_content in chapter_content.items():
                for _, style_content in paragraph_content.items():
                    for entry in style_content:
                        character = entry.get("character", "N/A")
                        sentence = entry.get("sent", "N/A")
                        timestamp = entry.get("time", "N/A")
                        recipient = entry.get("to whom", "N/A")
                        extracted_scripts.append(
                            {
                                "character": character,
                                "sent": sentence,
                                "time": timestamp,
                                "to_whom": recipient,
                            }
                        )
        return extracted_scripts


if __name__ == "__main__":
    yaml_filepath = (
        "/data/corpus/J-MAC/txt-fixed/楠山正雄/おおかみと七ひきのこやぎ/おおかみと七ひきのこやぎ (1)/all.yaml"
    )
    script_extractor = AudiobookScriptExtractor(yaml_filepath)
    reading_scripts = script_extractor.extract_reading_scripts()
    for script in reading_scripts:
        print(f"{script['character']}: {script['sent']}")
