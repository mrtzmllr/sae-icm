import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="src/poet/config.yaml")
    
    known, unknown = parser.parse_known_args()

    return known, unknown


def set_by_dotted_key(conf, key, value):
    keys = key.split(".")
    cur = conf
    for k in keys[:-1]:
        if k not in cur:
            raise ValueError(f"Wrong key: {k} at configuration {cur}")
        cur = cur[k]
    cur[keys[-1]] = value


def handle_unknown_args(unknown, conf):
    it = iter(unknown)
    for key in it:
        print("\n", "key\n", key)
        if not key.startswith("--"):
            raise ValueError(f"Unexpected argument {key}")
        value = next(it)

        dotted_key = key[2:]
        parsed_value = yaml.safe_load(value) # preserves int/float/bool

        set_by_dotted_key(conf, dotted_key, parsed_value)

    return conf