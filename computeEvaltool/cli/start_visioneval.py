import os
from argparse import ArgumentParser

from computeEvaltool.cli.base import CLICommand


def subparser_func(args):
    """ Function which will be called for a specific sub parser.
    """
    return VisionBenchCMD(args)


class VisionBenchCMD(CLICommand):
    name = 'visioneval'

    def __init__(self, args):
        self.args = args

    @staticmethod
    def define_args(parsers: ArgumentParser):
        """ define args for create pipeline template command.
        """
        from computeEvaltool.visioneval.argument import add_argument

        parser = parsers.add_parser(
            VisionBenchCMD.name,
            help='Run VisionEval performance benchmark.'
        )
        add_argument(parser)
        parser.set_defaults(func=subparser_func)

    def execute(self):
        from computeEvaltool.visioneval.main import run_vision_benchmark

        run_vision_benchmark(self.args)