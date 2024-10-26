"""
Platform-agnostic test harness, with ABC for test scaffold and asyncio-based
test management.
"""

import asyncio
import json
from os import makedirs
from os.path import exists
from abc import ABC, abstractmethod


class AbstractTestScaffold(ABC):
    """ABC for test scaffold"""

    @abstractmethod
    def setup(self, test_case):
        """Setup code before test case is run (typically for subclass state)."""

    @abstractmethod
    def run_test_case(self, test_case, environment):
        """Run the test case end-to-end; return a number encoding the points allocated."""


def run_test(scaffold, test_case):
    """Ran a single test case with the scaffold; returns score."""
    environment = scaffold.setup(test_case)
    try:
        return scaffold.run_test_case(test_case, environment)
    except Exception as exception:  # pylint: disable=broad-except
        print(f"Exception during test: {exception}")
        return 0


async def run_test_wrapper(interpreter, test_case, timeout):
    """
    Wrapper for run_test with timeout and minor debugging.
    Uses asyncio to enforce timeout, not for concurrency.
    """
    print(f'Running {test_case["srcfile"]}... ', end="")
    try:
        async with asyncio.timeout(timeout):
            result = await asyncio.to_thread(run_test, interpreter, test_case)
            print(f' {"PASSED" if result else "FAILED"}')
            return result
    except asyncio.TimeoutError:
        print("TIMED OUT")
        return 0


async def run_all_tests(interpreter, tests, timeout_per_test=5, zero_credit=False):
    """
    Run all tests sequentially; defaults to 5s timeout per test.
    Each test case *must* have a name and srcfile key.
    """
    print(f"Running {len(tests)} tests...")
    results = [
        {
            "name": test["name"],
            "score": await run_test_wrapper(interpreter, test, timeout_per_test) if not zero_credit else 0,
            "max_score": 1,
            "visibility": "visible"
            if test.get("visible", False)
            else "after_published",
        }
        for test in tests
    ]
    print(f"{get_score(results)}/{len(tests)} tests passed.")
    return results


def format_gradescope_output(results):
    """Generate proper JSON object depending on results type."""
    if isinstance(results, (int, float)):
        return {"score": results}
    return {"tests": results}


def write_gradescope_output(score, is_prod):
    """Write a results.json with the score; use CWD on dev, root on prod."""
    path = "/autograder/results" if is_prod else "."
    data = format_gradescope_output(score)
    if not exists(path):
        print(f"{path} does not exist, creating...")
        makedirs(path)
    with open(f"{path}/results.json", "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=4)

def write_gradescope_output_failure(msg, is_prod):
    """Used if the submission code cannot launch e.g., due to syntax error or missing file"""
    results = [{
        "score": 0,
        "status": "failed",
        "name": "Pre-launch check",
        "output": msg,
    }]

    write_gradescope_output(results, is_prod)

def get_score(results):
    """Helper to get student's score (for 0/1-based scores.)"""
    return len(list(filter(lambda result: result["score"], results)))