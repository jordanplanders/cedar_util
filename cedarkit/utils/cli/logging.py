import datetime
import sys
import time

#
# def print_log_line(script, function, log_line, level=0, log_type='info'):
#     if log_type == 'error':
#         file_pointer = sys.stderr
#     else:
#         file_pointer = sys.stdout
#
#     if isinstance(log_line, list):
#         log_line = ', '.join(log_line)
#
#     tab_level = '\t' * level
#
#     timestamp = datetime.datetime.now()
#     print(timestamp.strftime('%Y-%m-%d %H:%M:%S'), f'{tab_level}{script}: {function}', log_line , file=file_pointer, flush=True)
#     return time.time()


# cedar_utils/log_utils.py
import logging
import os
import sys


def get_log_level(env_var: str = "CEDAR_LOG_LEVEL", default: str = "INFO") -> int:
    """Map env var to a logging level, with a safe default."""
    name = os.getenv(env_var, default).upper()
    return getattr(logging, name, logging.INFO)


class StdoutFilter(logging.Filter):
    """Allow only records below ERROR (DEBUG/INFO/WARNING)."""
    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        return record.levelno < logging.ERROR


class StderrFilter(logging.Filter):
    """Allow only ERROR and CRITICAL."""
    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        return record.levelno >= logging.ERROR


def setup_logging(
    env_var: str = "CEDAR_LOG_LEVEL",
    default_level: str = "INFO",
    force: bool = False,
) -> None:
    """Configure the root logger once, splitting output by level across streams.

    Sets the root logger's level from an environment variable, and attaches
    two stream handlers: one for stdout carrying everything below ERROR
    (DEBUG/INFO/WARNING), one for stderr carrying ERROR and CRITICAL. A
    no-op if the root logger already has handlers and ``force`` is
    ``False``, so it's safe to call this at the top of every module without
    duplicating handlers.

    Parameters
    ----------
    env_var : str, default 'CEDAR_LOG_LEVEL'
        Environment variable read for the log level name (e.g. ``'DEBUG'``).
    default_level : str, default 'INFO'
        Level name used if ``env_var`` isn't set, or doesn't match a real
        logging level.
    force : bool, default False
        If ``True``, remove any existing handlers on the root logger (e.g.
        ones added by Jupyter) and reconfigure from scratch. If ``False``
        and handlers are already present, this function does nothing.

    See Also
    --------
    log_line : Emits a log message once logging is configured.
    """
    root = logging.getLogger()

    if root.handlers and not force:
        # Already configured; don't double-add handlers
        return

    if force:
        # Strip any existing handlers (Jupyter, etc.)
        for h in root.handlers[:]:
            root.removeHandler(h)

    level = get_log_level(env_var, default_level)
    root.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # stdout handler for DEBUG/INFO/WARNING
    h_out = logging.StreamHandler(stream=sys.stdout)
    h_out.setLevel(level)
    h_out.addFilter(StdoutFilter())
    h_out.setFormatter(formatter)

    # stderr handler for ERROR/CRITICAL
    h_err = logging.StreamHandler(stream=sys.stderr)
    h_err.setLevel(level)
    h_err.addFilter(StderrFilter())
    h_err.setFormatter(formatter)

    root.addHandler(h_out)
    root.addHandler(h_err)


def log_line(
    logger: logging.Logger,
    log_line,
    *,
    indent: int = 0,
    log_type: str = "info",
) -> None:
    """Log a message (or list of messages) through ``logger``, tab-indented.

    A thin wrapper that joins list messages with commas, prefixes the
    result with ``indent`` tab characters, and dispatches to the named
    level method on ``logger`` — used throughout CedarKit in place of the
    module's earlier ``print``-based logging.

    Parameters
    ----------
    logger : logging.Logger
        Logger to write to (typically a module- or class-level logger).
    log_line : str or list
        Message to log. If a list, elements are stringified and joined with
        ``', '``.
    indent : int, default 0
        Number of tab characters to prefix the message with.
    log_type : {'debug', 'info', 'warning', 'error', 'critical'}, default 'info'
        Logger method to call. Falls back to ``logger.info`` if the name
        isn't a valid method.

    See Also
    --------
    setup_logging : Configures the handlers this function's output goes to.
    """
    if isinstance(log_line, list):
        log_line = ", ".join(map(str, log_line))

    prefix = "\t" * indent
    msg = f"{prefix}{log_line}"

    log_method = getattr(logger, log_type, logger.info)
    log_method(msg)
