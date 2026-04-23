import json
import os
import re


class ConfigError(Exception):
    pass


def _strip_trailing_commas(raw_config):
    return re.sub(r",(\s*[}\]])", r"\1", raw_config)


def _has_openai_config(config):
    return any(
        str(config.get(key, "")).strip() for key in ("OPENAI_MODEL", "OPENAI_API_BASE")
    )


def _normalize_config(config):
    normalized = dict(config)
    if "SHAI_API_PROVIDER" not in normalized and _has_openai_config(config):
        normalized["SHAI_API_PROVIDER"] = "openai"
    return normalized


def debug_print(*args, **kwargs):
    if os.environ.get("DEBUG", "").lower() == "true":
        print(*args, **kwargs)


def load_config():
    # Determine the platform
    platform = os.name  # posix, nt, java, etc.
    config_app_name = "shell-ai"

    # Default configuration values
    default_config = {
        "OPENAI_MODEL": "gpt-3.5-turbo",
        "SHAI_SUGGESTION_COUNT": "3",
        "SHAI_API_PROVIDER": "groq",
        "GROQ_MODEL": "llama-3.3-70b-versatile",
        "SHAI_TEMPERATURE": "0.05",
    }

    try:
        # Determine the path to the configuration file based on the platform
        if platform == "posix":
            config_path = os.path.expanduser(f"~/.config/{config_app_name}/config.json")
        elif platform == "nt":
            config_path = os.path.join(
                os.environ["APPDATA"], config_app_name, "config.json"
            )
        else:
            raise Exception("Unsupported platform")

        debug_print(f"Looking for config file at: {config_path}")

        # Read the configuration file
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = f.read()

        try:
            parsed_config = json.loads(raw_config)
        except json.JSONDecodeError as exc:
            sanitized_config = _strip_trailing_commas(raw_config)
            if sanitized_config == raw_config:
                raise ConfigError(
                    f"Invalid JSON in config file at {config_path}: {exc.msg}"
                ) from exc
            try:
                parsed_config = json.loads(sanitized_config)
            except json.JSONDecodeError as sanitized_exc:
                raise ConfigError(
                    f"Invalid JSON in config file at {config_path}: {sanitized_exc.msg}"
                ) from sanitized_exc

        config = _normalize_config(parsed_config)
        debug_print("Found and loaded config file successfully")

        # Merge with defaults, keeping user settings where they exist
        return {**default_config, **config}
    except FileNotFoundError:
        debug_print("No config file found, using default configuration")
        return default_config
    except ConfigError:
        raise
    except Exception as e:
        debug_print(
            f"Unexpected error loading config: {str(e)}, using default configuration"
        )
        return default_config
