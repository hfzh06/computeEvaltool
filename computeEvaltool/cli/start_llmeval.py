import os
from argparse import ArgumentParser

from computeEvaltool.cli.base import CLICommand


def subparser_func(args):
    """ Function which will be called for a specific sub parser.
    """
    return PerfBenchCMD(args)


class PerfBenchCMD(CLICommand):
    name = 'llmeval'

    def __init__(self, args):
        self.args = args

    @staticmethod
    def define_args(parsers: ArgumentParser):
        """ define args for create pipeline template command.
        """
        from computeEvaltool.llmeval.arguments import add_argument

        parser = parsers.add_parser(PerfBenchCMD.name)
        add_argument(parser)
        parser.set_defaults(func=subparser_func)

    def execute(self):
        from computeEvaltool.llmeval.main import run_perf_benchmark

        run_perf_benchmark(self.args)
