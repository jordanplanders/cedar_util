import logging
from pathlib import Path

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
        return run_continue, overwrite

    # 3) otherwise → run & overwrite
    return run_continue, overwrite





def setup_logging(script_name, log_destination=None):
    """
    Configures logging to either a file or the console based on the argument.

    Args:
        script_name (str): The name of the script being logged.
        log_destination (str or None): Path to the log file. If None, logs to the console.
    """
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the logging level

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set the logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if log_destination:
        # Log to a file
        file_handler = logging.FileHandler(log_destination)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging started for script: {script_name}. Logs are being written to {log_destination}.")
    else:
        # Log to the console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.info(f"Logging started for script: {script_name}. Logs are being printed to the console.")

# Example usage
if __name__ == "__main__":
    script_name = Path(__file__).name  # Get the current script's filename
    log_file = "output.log"  # Specify the log file name
    setup_logging(script_name, log_file)
    logging.info("This is a log message.")