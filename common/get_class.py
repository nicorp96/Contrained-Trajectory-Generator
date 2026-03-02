import importlib


def get_class_dict(config: dict):
    class_path = config["target"]
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    class_mdl = getattr(module, class_name)
    return class_mdl(config)


def get_class_str(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
