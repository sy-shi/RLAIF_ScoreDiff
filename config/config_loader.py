import collections.abc
import os
import pathlib
import yaml

from callbacks import LoggingCallbacks
from ray import tune
import pdb


class ConfigLoader():

    def load_config(name, use_hpo = False):
        names = name.split("/")

        path = os.path.join("config", names[0] + ".yaml")
        
        configs = yaml.safe_load(open(path))

        configs = ConfigLoader._process_config(configs)
        configs = ConfigLoader._initialize_configs(configs)

        config = configs["BASE_CONFIG"]

        if len(names) > 1:
            config.update(configs[names[1]])

        if "HPO_CONFIG" in configs and use_hpo:
            # config.update(configs["HPO_CONFIG"])
            config = ConfigLoader._update_config(config, configs["HPO_CONFIG"])

        return config
    load_config = staticmethod(load_config)


    def _process_config(config):
        for key, value in config.items():
            if isinstance(value, dict):
                config[key] = ConfigLoader._process_config(value)

            elif isinstance(value, str):
                if len(value) >= 5 and value[:5] == "tune.":
                    config[key] = eval(value)

                elif len(value) >= 5 and value[:5] == "$SRC/":
                    # Note: could use string replace here instead of only checking prefix but am not sure it would ever be necessary
                    config[key] = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), value[5:])

        return config
    _process_config = staticmethod(_process_config)


    def _initialize_configs(config):
        # config["ENV_CONFIG"]["model"] = config["MODEL_CONFIG"]

        config["BASE_CONFIG"]["env_config"] = config["ENV_CONFIG"]
        config["BASE_CONFIG"]["model"] = config["MODEL_CONFIG"]
        config["BASE_CONFIG"]["callbacks"] = LoggingCallbacks

        return config
    _initialize_configs = staticmethod(_initialize_configs)


    def _update_config(d, u):
        for k, v in u.items():
            if isinstance(d, collections.abc.Mapping):
                if isinstance(v, collections.abc.Mapping):
                    r = ConfigLoader._update_config(d.get(k, {}), v)
                    d[k] = r
                else:
                    d[k] = u[k]
            else:
                d = {k: u[k]}
        return d
    _update_config = staticmethod(_update_config)
