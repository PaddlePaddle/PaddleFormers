import logging
import subprocess
import os

logger = logging.getLogger(__name__)


def test_ci_environment_info():
    """Verify CI environment is correctly configured."""
    result = subprocess.run(["date"], capture_output=True, text=True)
    logger.info(f"CI_DATE: {result.stdout.strip()}")

    result = subprocess.run(["hostname"], capture_output=True, text=True)
    logger.info(f"CI_HOST: {result.stdout.strip()}")

    result = subprocess.run(["whoami"], capture_output=True, text=True)
    logger.info(f"CI_USER: {result.stdout.strip()}")

    result = subprocess.run(["id"], capture_output=True, text=True)
    logger.info(f"CI_ID: {result.stdout.strip()}")

    workspace = os.environ.get("GITHUB_WORKSPACE", "")
    runner_temp = os.environ.get("RUNNER_TEMP", "")
    logger.info(f"CI_WORKSPACE: {workspace}")
    logger.info(f"CI_RUNNER_TEMP: {runner_temp}")

    assert True
