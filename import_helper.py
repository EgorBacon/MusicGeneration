import importlib
from pathlib import Path
import sys


def import_submodule(module_name):
    parts = module_name.split(".")
    package_name = parts[0]

    # Create a placeholder for the top-level package
    package_spec = importlib.util.find_spec(package_name)
    sys.modules[package_name] = importlib.util.module_from_spec(package_spec)

    # Create the submodule
    submodule_path = Path(package_spec.submodule_search_locations[0]) / Path(*parts[1:])
    if submodule_path.is_dir():
        submodule_path = submodule_path / "__init__.py"
    else:
        submodule_path = submodule_path.with_suffix(".py")
    submodule_spec = importlib.util.spec_from_file_location(module_name, submodule_path)
    submodule_module = importlib.util.module_from_spec(submodule_spec)
    sys.modules[module_name] = submodule_module

    # Load the submodule
    submodule_spec.loader.exec_module(submodule_module)

    return submodule_module
