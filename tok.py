import miditok
import json

path = "/Users/paultriana/creative_labs/MMM/configs/MMM.json"
save_path = "/Users/paultriana/creative_labs/MMM/configs/MMM_update.json"

with open(path, 'r') as f:
    config_data = json.load(f)

config_data["config"]["time_signature_range"] = { int(k): v for k, v in config_data["config"]["time_signature_range"].items() }
for beat_res_key in ["beat_res", "beat_res_rest"]:
    config_data["config"][beat_res_key] = {
        tuple(map(int, beat_range.split("_"))): res
        for beat_range, res in config_data["config"][beat_res_key].items()
    }
config = miditok.TokenizerConfig.from_dict(config_data["config"])
tokenizer = miditok.MMM(config)
tokenizer.config.save_to_json(save_path)
config2 = miditok.TokenizerConfig.load_from_json(save_path)
tokenizer2 = miditok.MMM(config2)

tokenizer3 = miditok.MMM(None, path)
tokenizer3.save(save_path)