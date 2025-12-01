# conftest.py  -- Enterprise Enhanced Logging System (Final Optimized Edition)
import logging
import os
import sys
import json
import pytest
from datetime import datetime
from multiprocessing import Queue
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pythonjsonlogger import jsonlogger
import colorlog


###############################################################################
# 1. worker log queue
###############################################################################
LOG_QUEUE = Queue()


###############################################################################
# 2. calor formatter
###############################################################################
def build_color_formatter():
    return colorlog.ColoredFormatter(
        "%(log_color)s[%(asctime)s] [%(worker)s] [%(levelname)s] "
        "[%(test_case)s]%(reset)s %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "white",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )


###############################################################################
# 3. JSON log formatter
###############################################################################
class JSONLogFormatter(jsonlogger.JsonFormatter):
    def process_log_record(self, log_record):
        # add utc timestamp
        log_record["timestamp"] = datetime.utcnow().isoformat() + "Z"
        return super().process_log_record(log_record)


###############################################################################
# 4.  worker logger
###############################################################################
def create_worker_logger(worker_id):
    logger = logging.getLogger(f"pf_worker_{worker_id}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # 
    if not logger.handlers:
        qh = QueueHandler(LOG_QUEUE)
        logger.addHandler(qh)

    return logger


###############################################################################
# 5. master pytest_all.log / pytest_error.log listener
###############################################################################
def setup_master_log_listener():
    logs_dir = "pytest_logs"
    os.makedirs(logs_dir, exist_ok=True)

    #  pytest_all.log
    rotate_main = RotatingFileHandler(
        f"{logs_dir}/pytest_all.log",
        maxBytes=20 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    rotate_main.setLevel(logging.DEBUG)
    rotate_main.setFormatter(JSONLogFormatter())

    #  pytest_error.log
    rotate_err = RotatingFileHandler(
        f"{logs_dir}/pytest_error.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    rotate_err.setLevel(logging.ERROR)
    rotate_err.setFormatter(JSONLogFormatter())

    listener = QueueListener(
        LOG_QUEUE,
        rotate_main,
        rotate_err,
        respect_handler_level=True,
    )
    listener.start()
    return listener


###############################################################################
# 6. each test file logger
###############################################################################
def setup_per_test_file_logger(file_path, worker_id):
    logs_dir = "pytest_logs"
    os.makedirs(logs_dir, exist_ok=True)

    file_name = os.path.basename(file_path).replace(".py", ".log")
    log_path = f"{logs_dir}/{file_name}"

    fh = RotatingFileHandler(log_path, maxBytes=3 * 1024 * 1024, backupCount=3)
    fh.setFormatter(JSONLogFormatter())
    fh.setLevel(logging.DEBUG)

    logger = logging.getLogger(f"pf_file_{file_name}_{worker_id}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    #  handler
    if not logger.handlers:
        logger.addHandler(fh)

    return logger


###############################################################################
# 7. pytest initialize logging
###############################################################################
def pytest_configure(config):
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")

    # 
    config._pf_logger = create_worker_logger(worker_id)

    #  pytest_all.log/pytest_error.log
    if worker_id == "master":
        config._pf_listener = setup_master_log_listener()

        #  console 
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(build_color_formatter())

        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

        # 
        if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
            root.addHandler(console)

    print(f"[{worker_id}] Enhanced logging initialized.")


###############################################################################
# 8. test context (logger adapter)
###############################################################################
@pytest.fixture(autouse=True)
def inject_test_context(request):
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")
    logger = request.config._pf_logger

    # Test 
    test_name = request.node.name
    cls_name = request.node.cls.__name__ if request.node.cls else "Module"
    full_name = f"{cls_name}.{test_name}"

    # logger adapter
    adapter = logging.LoggerAdapter(
        logger,
        {"test_case": full_name, "worker": worker_id},
    )

    # log file
    file_logger = setup_per_test_file_logger(request.fspath.strpath, worker_id)
    file_adapter = logging.LoggerAdapter(
        file_logger,
        {"test_case": full_name, "worker": worker_id},
    )

    request.node._logger = adapter
    request.node._file_logger = file_adapter

    yield


###############################################################################
# 9. log test start / end / failures
###############################################################################
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    item._start_time = datetime.now()
    item._logger.info("TEST START")
    item._file_logger.info("TEST START")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    duration = (datetime.now() - item._start_time).total_seconds()
    msg = f"TEST END | duration={duration:.3f}s"
    item._logger.info(msg)
    item._file_logger.info(msg)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()

    if rep.failed:
        msg = f"FAILED | {rep.longreprtext}"
        item._logger.error(msg)
        item._file_logger.error(msg)


###############################################################################
# 10. test session 
###############################################################################
def pytest_sessionfinish(session, exitstatus):
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")

    if worker_id == "master":
        listener = session.config._pf_listener
        listener.stop()

        print("\n========== LOG SUMMARY ==========")
        print("pytest_logs/pytest_all.log    (JSON structured)")
        print("pytest_logs/pytest_error.log  (only ERROR/CRITICAL)")
        print("pytest_logs/test_xxx.log per test file")
        print("=================================\n")
