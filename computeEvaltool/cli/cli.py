import argparse

from computeEvaltool import __version__
from computeEvaltool.cli.start_llmeval import PerfBenchCMD
from computeEvaltool.cli.start_visioneval import VisionBenchCMD

def run_cmd():
    parser = argparse.ArgumentParser('computeEvaltool Command Line tool', usage='computeEvaltool <command> [<args>]')
    parser.add_argument('-v', '--version', action='version', version=f'computeEvaltool {__version__}')
    subparsers = parser.add_subparsers(help='computeEvaltool command line helper.')

    PerfBenchCMD.define_args(subparsers)
    VisionBenchCMD.define_args(subparsers)


    args = parser.parse_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        exit(1)

    cmd = args.func(args)
    cmd.execute()


if __name__ == '__main__':
    run_cmd()