import importlib


def get_class_dict(config: dict, val=False):
    class_path = config["target"]
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    class_mdl = getattr(module, class_name)
    if val:
        return class_mdl(config, val=val)
    return class_mdl(config)


def get_class_str(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
