# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# The CLI entrypoint to vLLM.
import signal
import sys

import vllm.entrypoints.cli.benchmark.main
import vllm.entrypoints.cli.collect_env
import vllm.entrypoints.cli.openai
import vllm.entrypoints.cli.run_batch
import vllm.entrypoints.cli.serve
import vllm.version
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup
from vllm.utils import FlexibleArgumentParser

CMD_MODULES = [
    vllm.entrypoints.cli.openai,
    vllm.entrypoints.cli.serve,
    vllm.entrypoints.cli.benchmark.main,
    vllm.entrypoints.cli.collect_env,
    vllm.entrypoints.cli.run_batch,
]


def register_signal_handlers():

    def signal_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)


def main():
    cli_env_setup()

    parser = FlexibleArgumentParser(
        description="vLLM CLI",
        epilog=VLLM_SUBCMD_PARSER_EPILOG,
    )
    parser.add_argument('-v',
                        '--version',
                        action='version',
                        version=vllm.version.__version__)
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}
    for cmd_module in CMD_MODULES:
        # <module 'vllm.entrypoints.cli.openai' from '/Users/bytedance/ly/code/vllm/vllm/entrypoints/cli/openai.py'>
        # <module 'vllm.entrypoints.cli.serve' from '/Users/bytedance/ly/code/vllm/vllm/entrypoints/cli/serve.py'>
        # <module 'vllm.entrypoints.cli.benchmark.main' from '/Users/bytedance/ly/code/vllm/vllm/entrypoints/cli/benchmark/main.py'>
        # <module 'vllm.entrypoints.cli.collect_env' from '/Users/bytedance/ly/code/vllm/vllm/entrypoints/cli/collect_env.py'>
        new_cmds = cmd_module.cmd_init()
        # [<vllm.entrypoints.cli.openai.ChatCommand object at 0x144400f70>, <vllm.entrypoints.cli.openai.CompleteCommand object at 0x1461b3520>]
        # [<vllm.entrypoints.cli.serve.ServeSubcommand object at 0x1591ce9a0>]
        # [<vllm.entrypoints.cli.benchmark.main.BenchmarkSubcommand object at 0x13f958c70>]
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(
                dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd
    args = parser.parse_args()
    if args.subparser in cmds:
        cmds[args.subparser].validate(args)

    # 上面for循环中会设置dispatch_function
    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
