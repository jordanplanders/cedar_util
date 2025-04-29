import os
import sys
from pathlib import Path

from utils.arg_parser import get_parser
from utils.config_parser import load_config
from utils.location_helpers import *

# python3 carc2/make_dirs.py --project $PROJECT

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    # Example usage of arguments
    # print(f"Number from script2: {args.number}")
    if args.project is not None:
        proj_name = args.project
    else:
        print('project name is required', file=sys.stdout, flush=True)
        print('project name is required', file=sys.stderr, flush=True)
        sys.exit(0)

    # current_path = Path(os.getcwd())
    # if proj_name in str(current_path):
    #     proj_dir = Path(str(current_path).split(proj_name)[0])/ proj_name
    # else:
    #     proj_dir = current_path / proj_name
    #
    proj_dir = set_proj_dir(proj_name, Path(os.getcwd()))

    config = load_config(proj_dir / 'proj_config.yaml')
    carc_config_d = config.calc_carc_dir

    (proj_dir / 'parameters').mkdir(parents=True, exist_ok=True)  # Create the parameters directory if it does not exist
    (proj_dir/ carc_config_d.name).mkdir(parents=True, exist_ok=True)  # Create the parameters directory if it does not exist

    calc_location = set_calc_path(args, proj_dir, config)
    calc_location.mkdir(parents=True, exist_ok=True)  # Create the parameters directory if it does not exist

    (proj_dir/'surrogates').mkdir(parents=True, exist_ok=True)  # Create the parameters directory if it does not exist
    (proj_dir/'slurm').mkdir(parents=True, exist_ok=True)  # Create the parameters directory if it does not exist
    (proj_dir/'notebooks').mkdir(parents=True, exist_ok=True)  # Create the parameters directory if it does not exist

    (calc_location/carc_config_d.dirs.calc_convergence_dir).mkdir(parents=True, exist_ok=True)  # Create the parameters directory if it does not exist
    (calc_location/carc_config_d.dirs.output).mkdir(parents=True, exist_ok=True)  # Create the parameters directory if it does not exist

