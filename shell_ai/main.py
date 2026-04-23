#!/usr/bin/env python3
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import time
from enum import Enum

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from langchain.schema import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI

from shell_ai.code_parser import ContextManager, code_parser
from shell_ai.config import load_config
from shell_ai.parallel_suggestions import generate_suggestions_parallel


class SelectSystemOptions(Enum):
    OPT_GEN_SUGGESTIONS = "Generate new suggestions"
    OPT_DISMISS = "Dismiss"
    OPT_NEW_COMMAND = "Enter a new command"


class APIProvider(Enum):
    openai = "openai"
    azure = "azure"
    groq = "groq"
    ollama = "ollama"
    mistral = "mistral"


class Colors:
    WARNING = "\033[93m"
    END = "\033[0m"


def debug_print(*args, **kwargs):
    if os.environ.get("DEBUG", "").lower() == "true":
        print(*args, **kwargs)


def get_active_shell_name():
    if platform.system() == "Windows":
        if os.environ.get("PSModulePath"):
            return "powershell"
        return "cmd"

    shell = os.environ.get("SHELL", "")
    if shell:
        return os.path.basename(shell)
    return "sh"


def get_history_config(shell_name):
    history_file_path = None
    history_format = None

    if shell_name == "zsh":
        history_file_path = os.path.expanduser("~/.zsh_history")
        history_format = ": {}:0;{}\n"
    elif shell_name == "bash":
        history_file_path = os.path.expanduser("~/.bash_history")
        history_format = ": {}:0;{}\n"
    elif shell_name in ("csh", "tcsh"):
        history_file_path = os.path.expanduser("~/.history")
        history_format = "{} {}\n"
    elif shell_name == "ksh":
        history_file_path = os.path.expanduser("~/.sh_history")
        history_format = ": {}:0;{}\n"
    elif shell_name == "fish":
        history_file_path = os.path.expanduser("~/.local/share/fish/fish_history")
        history_format = "- cmd: {}\n  when: {}\n"
    elif shell_name == "powershell":
        appdata = os.environ.get("APPDATA", "")
        history_candidates = [
            os.path.join(
                appdata,
                "Microsoft",
                "PowerShell",
                "PSReadLine",
                "ConsoleHost_history.txt",
            ),
            os.path.join(
                appdata,
                "Microsoft",
                "Windows",
                "PowerShell",
                "PSReadLine",
                "ConsoleHost_history.txt",
            ),
        ]
        history_file_path = next(
            (
                path
                for path in history_candidates
                if os.path.isdir(os.path.dirname(path))
            ),
            history_candidates[-1],
        )
        history_format = "{}\n"

    return history_file_path, history_format


def write_command_history(shell_name, user_command):
    history_file_path, history_format = get_history_config(shell_name)
    if not history_file_path:
        return False

    os.makedirs(os.path.dirname(history_file_path), exist_ok=True)
    timestamp = int(time.time())
    with open(history_file_path, "a", encoding="utf-8") as history_file:
        history_file.write(history_format.format(timestamp, user_command))
    return True


def run_shell_command(user_command, capture_output=False):
    active_shell = get_active_shell_name()
    run_kwargs = {
        "check": True,
        "capture_output": capture_output,
        "text": capture_output,
    }

    if platform.system() == "Windows" and active_shell == "powershell":
        powershell_path = shutil.which("powershell") or shutil.which("pwsh")
        if not powershell_path:
            raise RuntimeError("PowerShell executable not found")
        return subprocess.run(
            [powershell_path, "-NoLogo", "-NoProfile", "-Command", user_command],
            **run_kwargs,
        )

    return subprocess.run(user_command, shell=True, **run_kwargs)


def main():
    """
    Required environment variables:
    - OPENAI_API_KEY: Your OpenAI API key. You can find this on https://beta.openai.com/account/api-keys

    Allowed envionment variables:
    - OPENAI_MODEL: The name of the OpenAI model to use. Defaults to `gpt-3.5-turbo`.
    - OLLAMA_MODEL: The name of the Ollama model to use. Defaults to `phi3.5`.
    - SHAI_SUGGESTION_COUNT: The number of suggestions to generate. Defaults to 3.
    - SHAI_SKIP_CONFIRM: Skip confirmation of the command to execute. Defaults to false. Set to `true` to skip confirmation.
    - SHAI_SKIP_HISTORY: Skip writing selected command to shell history (currently supported shells are zsh, bash, csh, tcsh, ksh, and fish). Defaults to false. Set to `true` to skip writing.
    - CTX: Allow the assistant to keep the console outputs as context allowing the LLM to produce more precise outputs. IMPORTANT: the outputs will be sent to OpenAI through their API, be careful if any sensitive data. Default to false.
    - SHAI_TEMPERATURE: Controls randomness in the output. Lower values make output more focused and deterministic (default: 0.05).
    - OLLAMA_API_BASE: The Ollama endpoint to use (default: "http://localhost:11434/v1/").
    Additional required environment variables for Azure Deployments:
    - OPENAI_API_KEY: Your OpenAI API key. You can find this on https://beta.openai.com/account/api-keys
    - OPENAI_API_TYPE: "azure"
    - AZURE_API_BASE
    - AZURE_DEPLOYMENT_NAME
    """

    # Load env configuration
    loaded_config = load_config()
    # Load keys of the configuration into environment variables
    for key, value in loaded_config.items():
        os.environ[key] = str(value)

    # Dump environment variables for debugging
    debug_print("Environment variables:")
    for key, value in os.environ.items():
        debug_print(f"{key}={value}")

    # Dump loaded configuration for debugging
    debug_print("\nLoaded configuration:")
    for key, value in loaded_config.items():
        debug_print(f"{key}={value}")
    debug_print()

    if (
        os.environ.get("OPENAI_API_KEY") is None
        and os.environ.get("GROQ_API_KEY") is None
        and os.environ.get("MISTRAL_API_KEY") is None
    ):
        print(
            "Please set either the OPENAI_API_KEY, MISTRAL_API_KEY or GROQ_API_KEY environment variable."
        )
        print(
            "You can also create `config.json` under `~/.config/shell-ai/` to set the API key, see README.md for more information."
        )
        sys.exit(1)

    TEXT_EDITORS = ("vi", "vim", "emacs", "nano", "ed", "micro", "joe", "nvim")

    CTX = os.environ.get("CTX", "False")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx", action="store_true", help="Set context mode to True.")
    parser.add_argument("prompt", type=str, nargs="*", default=None)
    args = parser.parse_args()
    if args.ctx:
        prompt = args.prompt
        CTX = "True"
    else:
        # Consume all arguments after the script name as a single sentence
        prompt = " ".join(sys.argv[1:])

    OPENAI_MODEL = os.environ.get("OPENAI_MODEL", loaded_config.get("OPENAI_MODEL"))
    OLLAMA_MODEL = os.environ.get(
        "OLLAMA_MODEL", loaded_config.get("OLLAMA_MODEL", "phi3.5")
    )
    OPENAI_MAX_TOKENS = os.environ.get("OPENAI_MAX_TOKENS", None)
    OLLAMA_MAX_TOKENS = os.environ.get(
        "OLLAMA_MAX_TOKENS", loaded_config.get("OLLAMA_MAX_TOKENS", 1500)
    )
    OLLAMA_API_BASE = os.environ.get(
        "OLLAMA_API_BASE",
        loaded_config.get("OLLAMA_API_BASE", "http://localhost:11434/v1/"),
    )
    OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", None)
    # Mistral configuration
    MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
    MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", loaded_config.get("MISTRAL_MODEL"))
    MISTRAL_API_BASE = os.environ.get(
        "MISTRAL_API_BASE",
        loaded_config.get("MISTRAL_API_BASE", "https://api.mistral.ai/v1"),
    )

    OPENAI_ORGANIZATION = os.environ.get("OPENAI_ORGANIZATION", None)
    OPENAI_PROXY = os.environ.get("OPENAI_PROXY", None)
    SHAI_SUGGESTION_COUNT = int(
        os.environ.get(
            "SHAI_SUGGESTION_COUNT", loaded_config.get("SHAI_SUGGESTION_COUNT", "3")
        )
    )

    # required configs just for azure openai deployments (faster)
    SHAI_API_PROVIDER = os.environ.get(
        "SHAI_API_PROVIDER", loaded_config.get("SHAI_API_PROVIDER", "openai")
    )

    OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION", "2023-05-15")
    if SHAI_API_PROVIDER not in APIProvider.__members__:
        print(
            f"Your SHAI_API_PROVIDER is not valid. Please choose one of {APIProvider.__members__}"
        )
        sys.exit(1)

    AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_DEPLOYMENT_NAME", None)
    AZURE_API_BASE = os.environ.get("AZURE_API_BASE", None)
    if SHAI_API_PROVIDER == "azure" and AZURE_DEPLOYMENT_NAME is None:
        print(
            "Please set the AZURE_DEPLOYMENT_NAME environment variable to your Azure deployment name."
        )
        sys.exit(1)
    if SHAI_API_PROVIDER == "azure" and AZURE_API_BASE is None:
        print(
            "Please set the AZURE_API_BASE environment variable to your Azure API base."
        )
        sys.exit(1)

    # Groq configuration
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    GROQ_MODEL = os.environ.get("GROQ_MODEL", loaded_config.get("GROQ_MODEL"))
    if SHAI_API_PROVIDER == "groq" and not GROQ_API_KEY:
        print("Please set the GROQ_API_KEY environment variable to your Groq API key.")
        sys.exit(1)

    if SHAI_API_PROVIDER == "mistral" and not MISTRAL_API_KEY:
        print(
            "Please set the MISTRAL_API_KEY environment variable to your Mistral API key."
        )
        sys.exit(1)

    # End loading configuration

    # Get temperature from config or environment
    SHAI_TEMPERATURE = float(
        os.environ.get(
            "SHAI_TEMPERATURE", loaded_config.get("SHAI_TEMPERATURE", "0.05")
        )
    )

    # Initialize chat provider based on configuration
    if SHAI_API_PROVIDER == "openai":
        chat = ChatOpenAI(
            model_name=OPENAI_MODEL,
            openai_api_base=OPENAI_API_BASE,
            openai_organization=OPENAI_ORGANIZATION,
            openai_proxy=OPENAI_PROXY,
            max_tokens=OPENAI_MAX_TOKENS,
            temperature=SHAI_TEMPERATURE,
        )
    elif SHAI_API_PROVIDER == "azure":
        chat = AzureChatOpenAI(
            openai_api_base=AZURE_API_BASE,
            openai_api_version=OPENAI_API_VERSION,
            deployment_name=AZURE_DEPLOYMENT_NAME,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_api_type="azure",
            temperature=SHAI_TEMPERATURE,
        )
    elif SHAI_API_PROVIDER == "groq":
        chat = ChatGroq(
            model_name=GROQ_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=SHAI_TEMPERATURE,
        )
    elif SHAI_API_PROVIDER == "ollama":
        chat = ChatOpenAI(
            model_name=OLLAMA_MODEL,
            openai_api_base=OLLAMA_API_BASE,
            max_tokens=OLLAMA_MAX_TOKENS,
            temperature=SHAI_TEMPERATURE,
            api_key="ollama",
        )
    elif SHAI_API_PROVIDER == "mistral":
        chat = ChatMistralAI(
            model_name=MISTRAL_MODEL,
            api_key=MISTRAL_API_KEY,
            base_url=MISTRAL_API_BASE,
            temperature=SHAI_TEMPERATURE,
        )

    active_shell = get_active_shell_name()

    if platform.system() == "Linux":
        info = platform.freedesktop_os_release()
        plaform_string = f"The system the shell command will be executed on is {platform.system()} {platform.release()}, running {info.get('ID')} version {info.get('VERSION_ID', info.get('BUILD_ID'))}. The active shell is {active_shell}. \n"
    else:
        plaform_string = f"The system the shell command will be executed on is {platform.system()} {platform.release()}. The active shell is {active_shell}. \n"

    def get_suggestions(prompt):
        base_system_message = (
            """You are an expert at using shell commands. I need you to provide a response in the format `{"command": "your_shell_command_here"}`. """
            + plaform_string
            + """ Only provide a single executable line of shell code as the value for the "command" key. Never output any text outside the JSON structure. The command will be directly executed in a shell. For example, if I ask to display the message abc, you should respond with ```json\n{"command": "echo abc"}\n```. Make sure the output is valid JSON."""
        )

        if active_shell == "powershell":
            base_system_message += (
                " Use PowerShell-compatible syntax, quoting, cmdlets, and pipelines."
                " Do not use bash-specific quoting or built-ins unless you explicitly invoke bash."
            )

        ctx = ContextManager.get_ctx()
        if ctx:
            base_system_message += (
                """ Between [], these are the last 1500 tokens from the previous command's output, you can use them as context: ["""
                + ctx
                + """]"""
            )

        system_message = SystemMessage(content=base_system_message)

        def generate_single_suggestion():
            messages = [
                system_message,
                HumanMessage(
                    content=f"Generate a shell command that satisfies this user request: {prompt}"
                ),
            ]
            debug_print(f"Messages: {messages}")
            response = chat.generate(messages=[messages])
            try:
                debug_print(f"Response: {response.generations[0][0].message.content}")
                json_content = code_parser(response.generations[0][0].message.content)
                command_json = json.loads(json_content)
                command = command_json.get("command", "")
                return command if command else None
            except (json.JSONDecodeError, IndexError):
                return None

        # Generate suggestions in parallel with max 4 workers
        commands = generate_suggestions_parallel(
            generate_single_suggestion, count=SHAI_SUGGESTION_COUNT, max_workers=4
        )

        # Filter out None values and deduplicate
        commands = list(set(cmd for cmd in commands if cmd))

        return commands

    if prompt:
        if CTX == "True":
            print(
                f"{Colors.WARNING}WARNING{Colors.END} Context mode: datas will be sent to OpenAI, be careful if any sensitive datas...\n"
            )
            print(f">>> {os.getcwd()}")
        while True:
            options = get_suggestions(prompt)
            options.append(SelectSystemOptions.OPT_GEN_SUGGESTIONS.value)
            options.append(SelectSystemOptions.OPT_NEW_COMMAND.value)
            options.append(SelectSystemOptions.OPT_DISMISS.value)

            # Get the terminal width
            terminal_width = os.get_terminal_size().columns

            # For each option for the name value of the Choice,
            # wrap the text to the terminal width
            choices = [
                Choice(
                    name=textwrap.fill(option, terminal_width, subsequent_indent="  "),
                    value=option,
                )
                for option in options
            ]
            try:
                selection = inquirer.select(
                    message="Select a command:", choices=choices
                ).execute()

                try:
                    if selection == SelectSystemOptions.OPT_DISMISS.value:
                        sys.exit(0)
                    elif selection == SelectSystemOptions.OPT_NEW_COMMAND.value:
                        prompt = input("New command: ")
                        continue
                    elif selection == SelectSystemOptions.OPT_GEN_SUGGESTIONS.value:
                        continue
                    if os.environ.get("SHAI_SKIP_CONFIRM") != "true":
                        user_command = inquirer.text(
                            message="Confirm:", default=selection
                        ).execute()
                    else:
                        user_command = selection

                    # Write executed command to shell history for easy reuse.
                    if os.environ.get("SHAI_SKIP_HISTORY") != "true":
                        if not write_command_history(active_shell, user_command):
                            print(
                                f"{Colors.WARNING}Warning:{Colors.END} Unsupported shell. History will not be saved. Please set SHAI_SKIP_HISTORY to true to disable."
                            )

                    # Default mode
                    if CTX == "False":
                        run_shell_command(user_command)
                        break
                    # Context mode
                    elif user_command.startswith(TEXT_EDITORS):
                        run_shell_command(user_command)
                    elif user_command.startswith("cd"):
                        path = os.path.expanduser("/".join(user_command.split(" ")[1:]))
                        os.chdir(path)
                    else:
                        result = run_shell_command(
                            user_command, capture_output=True
                        ).stdout
                        if len(result) > 0:
                            print(f"\n{result}")
                        ContextManager.add_chunk(result)
                    prompt = input(f">>> {os.getcwd()}\nNew command: ")
                except Exception as e:
                    print(f"{Colors.WARNING}Error{Colors.END} executing command: {e}")
            except KeyboardInterrupt:
                print("Exiting...")
                sys.exit(0)
    else:
        print("Describe what you want to do as a single sentence. `shai <sentence>`")


if __name__ == "__main__":
    main()
