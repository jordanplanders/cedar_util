import sys
import logging
logger = logging.getLogger(__name__)

try:
    from cedarkit.utils.cli import setup_logging, log_line
except ImportError:
    # Fallback: imports when running as a package
    from utils.cli.logging import setup_logging, log_line

def decide_file_handling(args, file_exists: bool, modify_datetime=None) -> tuple[bool, bool]:
    """
    Decide (run_continue, overwrite) based on:
      - args.override       (bool)
      - args.write          ("append" or "replace")
      - args.datetime_flag  (optional datetime cutoff)
      - file_exists         (bool)
      - modify_datetime     (file’s mtime as datetime or None)
    """
    # default to running and overwriting
    run_continue = True
    overwrite    = True

    # 1) if the file exists & no override → maybe skip
    if file_exists and not args.override:
        if args.datetime_flag is not None:
            try:
                if modify_datetime >= args.datetime_flag:
                    # file is fresh/newer than cutoff → skip
                    run_continue = False
                    overwrite    = False
                    return run_continue, overwrite
            except Exception:
                # if compare fails, ignore and proceed
                pass
        else:
            # no datetime_flag → skip unconditionally
            run_continue = False
            overwrite    = False
            return run_continue, overwrite

    # 2) if file exists & user asked to append → run & append
    if file_exists and args.write == "append":
        run_continue = True
        overwrite    = False
        print("Appending to existing file.", file=sys.stdout, flush=True)
        return run_continue, overwrite

    # 3) otherwise → run & overwrite
    return run_continue, overwrite
