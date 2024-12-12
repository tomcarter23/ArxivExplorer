import pytest
import sys

if __name__ == "__main__":
    sys.exit(pytest.main(["./tests/e2e/", "-vvv", "--durations=0", "-s", "--log-cli-level=INFO",]))