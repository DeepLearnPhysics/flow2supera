#!/usr/bin/env python3
import flow2supera
import sys,os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('input_file',
                    type = str,
                    help = "input file from which to read and simulate an event.",
                    )
parser.add_argument('-o', '--output',
                    type = str,
                    help = "Output (LArCV) filename.")
parser.add_argument('-c', '--config',
                    type = str,
                    default = '',
                    help = "Configuration keyword or a file path (full or relative including the file name).",
                    )
parser.add_argument('-n', '--num_events',
                    type = int,
                    default = None,
                    help="number of events to process.")
parser.add_argument('-s', '--skip',
                    type = int,
                    default = 0,
                    help="number of first events to skip.")
parser.add_argument('-l', '--log_file',
                    type = str,
                    default = '',
                    help="the name of a log file to be created.")

args = parser.parse_args()

if os.path.isfile(args.output):
    print('Ouput file already exists:', args.output)
    print('Exiting')
    sys.exit(1)

if not args.config in flow2supera.config.list_config() and not os.path.isfile(args.config):
    print('Invalid configuration given:',args.config)
    print('The argument is not valid as a file path nor matched with any of')
    print('predefined config keys:',flow2supera.config.list_config())
    print('Exiting')
    sys.exit(2)

if not args.input_file:
    print('No input files given! Exiting')
    sys.exit(3)

output = args.output
input_files = args.input_file

flow2supera.utils.run_supera(out_file=args.output,
                             in_file=args.input_file,
                             config_key=args.config,
                             num_events=args.num_events,
                             num_skip=int(args.skip),
                             save_log=args.log_file,
                             )
