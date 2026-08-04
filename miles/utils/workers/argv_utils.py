import argparse
import dataclasses
from collections.abc import Callable
from typing import TypeVar

from miles.utils.pydantic_utils import FrozenStrictBaseModel

CONFIG_JSON_FLAG = "--config-json"

_ConfigT = TypeVar("_ConfigT", bound=FrozenStrictBaseModel)
_ArgsT = TypeVar("_ArgsT")


def config_to_argv(config: FrozenStrictBaseModel) -> list[str]:
    argv = [CONFIG_JSON_FLAG, config.model_dump_json()]

    parsed = parse_config_argv(type(config), argv)
    assert parsed == config, f"config argv roundtrip mismatch: {parsed!r} != {config!r}"
    return argv


def parse_config_argv(config_cls: type[_ConfigT], argv: list[str] | None) -> _ConfigT:
    parser = argparse.ArgumentParser()
    parser.add_argument(CONFIG_JSON_FLAG, required=True)
    args = parser.parse_args(argv)
    return config_cls.model_validate_json(args.config_json)


def render_cli_argv(
    args_obj: _ArgsT,
    *,
    make_parser: Callable[[], argparse.ArgumentParser],
    from_parsed: Callable[[argparse.Namespace], _ArgsT],
    required_argv: list[str] | None = None,
    derived_fields: frozenset[str] = frozenset(),
) -> list[str]:
    """Render *args_obj* back into a command line that parses into an equal object.

    Fields named in *derived_fields* are computed by the parser from other flags and
    have no faithful command-line spelling of their own, so they are not rendered.
    The roundtrip below is what proves that omitting them loses nothing.
    """

    def parse(argv: list[str]) -> _ArgsT:
        return from_parsed(make_parser().parse_args(argv))

    base_argv = list(required_argv or [])
    argv = base_argv + _render_cli_argv(args_obj, cli_defaults=parse(base_argv), derived_fields=derived_fields)

    parsed = parse(argv)
    if parsed != args_obj:
        # A default the parser derives from other flags (e.g. the PD load balance
        # method) only reveals itself once those flags are on the command line, so
        # rendering once against the bare defaults can leave it unspelled.
        argv = argv + _render_cli_argv(args_obj, cli_defaults=parsed, derived_fields=derived_fields)
        parsed = parse(argv)

    assert parsed == args_obj, f"cli argv roundtrip mismatch: {parsed!r} != {args_obj!r}"
    return argv


def _render_cli_argv(args_obj: _ArgsT, *, cli_defaults: _ArgsT, derived_fields: frozenset[str]) -> list[str]:
    argv: list[str] = []
    for field in dataclasses.fields(args_obj):
        if field.name in derived_fields:
            continue
        value = getattr(args_obj, field.name)
        if value == getattr(cli_defaults, field.name):
            continue

        flag = "--" + field.name.replace("_", "-")
        if isinstance(value, bool):
            assert value, f"{flag} cannot be rendered: the CLI only has a flag for the non-default value"
            argv.append(flag)
        elif isinstance(value, list):
            argv.append(flag)
            argv.extend(str(item) for item in value)
        elif isinstance(value, dict):
            argv.append(flag)
            argv.extend(f"{key}={item}" for key, item in value.items())
        else:
            argv.extend([flag, str(value)])
    return argv
