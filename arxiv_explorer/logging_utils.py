import logging


def setup_logging(
    level: str, logger: logging.Logger = logging.getLogger(__name__)
) -> None:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s\t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
