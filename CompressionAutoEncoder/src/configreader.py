
import yaml

class NotFoundException(Exception):
    pass

class Config(object):

    def __init__(self, config_obj, name):
        self._config = config_obj
        self._name = name

    def get_value(self, key, default_value=None):
        try:
            if "/" in key:
                keys = key.split("/")
                current_block = self._config
                for split_key in keys:
                    if split_key in current_block:
                        current_block = current_block[split_key]
                    else:
                        raise NotFoundException
                return current_block
            else:
                if key in self._config:
                    return self._config[key]
                else:
                    raise NotFoundException
        except NotFoundException:
            if default_value:
                return self.get_value(default_value)
            else:
                raise NotFoundException("This key was not found: {} for: {}".format(key, self._name))


    def get_int(self, key, default_value=None):
        return int(self.get_value(key, default_value))

    def get_float(self, key, default_value=None):
        return float(self.get_value(key, default_value))

    def get_string(self, key, default_value=None):
        return str(self.get_value(key, default_value=default_value))

    def get_bool(self, key, default_value=None):
        ret = self.get_value(key, default_value)
        if isinstance(ret, str):
            ret = ret.lower()
            return ret == "true"
        elif isinstance(ret, int):
            return ret == 1
        return bool(ret)

class ConfigReader(object):

    def __init__(self, config_file_path):
        with open(config_file_path, 'r') as stream:
            file = yaml.safe_load(stream)
            if "config" in file:
                self.config = file["config"]

        self.data = Config(self.config["data"], "data")
        self.model = Config(self.config["model"], "model")
        self.optimizer = Config(self.config["optimizer"], "optimizer")

