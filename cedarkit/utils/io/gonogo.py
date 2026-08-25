import logging
logger = logging.getLogger(__name__)

from cedarkit.utils.cli import setup_logging, log_line

def decide_file_handling(args, file_exists: bool, modify_datetime=None) -> tuple[bool, bool]:
    """Decide whether to run and whether to overwrite, for a possibly-existing output file.

    Precedence: if the file exists and ``args.override`` is falsy, either
    skip (when there's no ``args.datetime_flag`` cutoff, or the file is
    newer than that cutoff) or fall through to later rules (if the datetime
    comparison fails). If the file exists and ``args.write == 'append'``,
    run without overwriting. Otherwise, run and overwrite.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (see ``get_parser``); reads ``args.override``
        (bool), ``args.write`` (``'append'`` or other), and
        ``args.datetime_flag`` (an optional datetime cutoff).
    file_exists : bool
        Whether the target output file already exists.
    modify_datetime : datetime.datetime, optional
        The existing file's modification time, compared against
        ``args.datetime_flag`` when both are given.

    Returns
    -------
    tuple of (bool, bool)
        ``(run_continue, overwrite)`` — whether to proceed with the run, and
        if so, whether to overwrite the existing file rather than append.
    """
    # default to running and overwriting
    run_continue = True
    overwrite    = True

    log_line(
        logger,
        (
            "decide_file_handling input: "
            f"file_exists={file_exists}, override={getattr(args, 'override', None)}, "
            f"write={getattr(args, 'write', None)}, "
            f"datetime_flag={getattr(args, 'datetime_flag', None)}, "
            f"modify_datetime={modify_datetime}"
        ),
        log_type="info",
    )
    # 1) if the file exists & no override → maybe skip
    if file_exists and not args.override:
        if args.datetime_flag is not None:
            try:
                if modify_datetime >= args.datetime_flag:
                    # file is fresh/newer than cutoff → skip
                    run_continue = False
                    overwrite    = False
                    log_line(
                        logger,
                        "decision: skip existing file (override=False and modify_datetime >= datetime_flag)",
                        log_type="info",
                    )
                    return run_continue, overwrite
            except Exception:
                # if compare fails, ignore and proceed
                log_line(
                    logger,
                    "decision branch: datetime comparison failed; continuing to later rules",
                    log_type="warning",
                )
                pass
        else:
            # no datetime_flag → skip unconditionally
            run_continue = False
            overwrite    = False
            log_line(
                logger,
                "decision: skip existing file (override=False and no datetime_flag)",
                log_type="info",
            )
            return run_continue, overwrite

    # 2) if file exists & user asked to append → run & append
    if file_exists and args.write == "append":
        run_continue = True
        overwrite    = False
        log_line(logger, "decision: append to existing file", log_type="info")
        return run_continue, overwrite

    # 3) otherwise → run & overwrite
    log_line(logger, "decision: run with overwrite", log_type="info")
    return run_continue, overwrite
