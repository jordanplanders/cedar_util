import os
import sys
from pathlib import Path

from utils.arg_parser import get_parser
from utils.config_parser import load_config


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

    proj_dir = Path(os.getcwd()) / proj_name
    config = load_config(proj_dir / 'proj_config.yaml')
    carc_config_d = config.calc_carc_dir

    (proj_dir / 'parameters').mkdir(parents=True, exist_ok=True)  # Create the parameters directory if it does not exist
    (proj_dir/ carc_config_d.name).mkdir(parents=True, exist_ok=True)  # Create the parameters directory if it does not exist
    (proj_dir/config.calc_dir).mkdir(parents=True, exist_ok=True)  # Create the parameters directory if it does not exist
    (proj_dir/'surrogates').mkdir(parents=True, exist_ok=True)  # Create the parameters directory if it does not exist
    (proj_dir/'slurm').mkdir(parents=True, exist_ok=True)  # Create the parameters directory if it does not exist

    if Path('/Users/jlanders').exists() == True:
        calc_location = proj_dir / config.local.calc_carc  #'calc_local_tmp'
    else:
        calc_location = proj_dir / config.carc.calc_carc

    calc_location.mkdir(parents=True, exist_ok=True)  # Create the parameters directory if it does not exist
    (calc_location/carc_config_d.dirs.calc_areas_dir).mkdir(parents=True, exist_ok=True)  # Create the parameters directory if it does not exist
    (calc_location/carc_config_d.dirs.calc_grp_pctile_dir).mkdir(parents=True, exist_ok=True)  # Create the parameters directory if it does not exist
    (calc_location/carc_config_d.dirs.cross_corr_dir).mkdir(parents=True, exist_ok=True)  # Create the parameters directory if it does not exist
    (calc_location/carc_config_d.dirs.calc_convergence_dir).mkdir(parents=True, exist_ok=True)  # Create the parameters directory if it does not exist
    (calc_location/carc_config_d.dirs.calc_metrics_dir).mkdir(parents=True, exist_ok=True)  # Create the parameters directory if it does not exist
    (calc_location/carc_config_d.dirs.ccm_surr_plots_dir_raw).mkdir(parents=True, exist_ok=True)  # Create the parameters directory if it does not exist

