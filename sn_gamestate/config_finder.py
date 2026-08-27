# Importing hf_resolver here (at plugin discovery) registers the `${hf:...}` OmegaConf
# resolver before any config interpolation is evaluated. Keep this import above the class.
from sn_gamestate.track import hf_resolver  # noqa: F401  (registers the 'hf' resolver)


class ConfigFinder:
    config_package = "pkg://sn_gamestate.configs"
