"""
Implements all CS 131-related test logic; is entry-point for testing framework.
"""

import asyncio
import importlib
from os import environ, listdir, getcwd
import sys
import traceback
from operator import itemgetter

from harness import (
    AbstractTestScaffold,
    run_all_tests,
    get_score,
    write_gradescope_output,
    write_gradescope_output_failure,
)


class TestScaffold(AbstractTestScaffold):
    """Implement scaffold for Brewin' interpreter; load file, validate syntax, run testcase."""

    def __init__(self, interpreter_lib):
        self.interpreter_lib = interpreter_lib

    def setup(self, test_case):
        srcfile = itemgetter("srcfile")(
            test_case
        )

        with open(srcfile, encoding="utf-8") as handle:
            prog_lines = handle.readlines()

        inp = self.__extract_test_data(prog_lines, "IN")
        expected = self.__extract_test_data(prog_lines, "OUT")

        program = "\n".join(prog_lines)

        return {
            "expected": expected,
            "stdin": inp,
            "program": program,
        }

    def run_test_case(self, test_case, environment):
        expect_failure = itemgetter("expect_failure")(test_case)
        stdin, expected, program = itemgetter("stdin", "expected", "program")(
            environment
        )
        interpreter = self.interpreter_lib.Interpreter(False, stdin, False)
        try:
            interpreter.run(program)
        except Exception as exception:  # pylint: disable=broad-except
            if expect_failure:

                error_type, _ = interpreter.get_error_type_and_line()
                received = [f"{error_type}"]

                if received == expected:
                    return 1

                print("\nExpected error:")
                print(expected)
                print("\nReceived error:")
                print(received)

            print("\nException: ")
            print(exception)
            traceback.print_exc()
            return 0

        if expect_failure:
            print("\nExpected error:")
            print(expected)
            print("\nActual output:")
            print(interpreter.get_output())
            return 0

        passed = interpreter.get_output() == expected
        if not passed:
            print("\nExpected output:")
            print(expected)
            print("\nActual output:")
            print(interpreter.get_output())

        return int(passed)


    def __extract_test_data(self, program, tag):
        in_soln = False
        soln = []
        for line in program:
            if line.strip() == f"*{tag}*":
                in_soln = not in_soln
            elif in_soln:
                line = line.rstrip("\n")
                soln.append(line)
        return soln


def __generate_test_case_structure(
    cases, directory, category="", expect_failure=False, visible=lambda _: True
):
    return [
        {
            "name": f"{category} | {i}",
            "srcfile": f"{directory}{i}.br",
            "expect_failure": expect_failure,
            "visible": visible(f"test{i}"),
        }
        for i in cases
    ]


def __generate_test_suite(version, successes, failures):
    return __generate_test_case_structure(
        successes,
        f"v{version}/tests/",
        "Correctness",
        False,
    ) + __generate_test_case_structure(
        failures,
        f"v{version}/fails/",
        "Incorrectness",
        True,
    )

def __get_file_names(folder_path):
    files_in_folder = listdir(folder_path)
    filenames = [file.split(".")[0] for file in files_in_folder]
    return filenames


def generate_test_suite_v1():
    """wrapper for generate_test_suite for v1"""
    tests = __get_file_names(getcwd() + "/v1/tests/")
    fails = __get_file_names(getcwd() + "/v1/fails/")
    return __generate_test_suite(
        1,
        tests,
        fails,
    )

def generate_test_suite_v2():
    """wrapper for generate_test_suite for v2"""
    tests = __get_file_names(getcwd() + "/v2/tests/")
    fails = __get_file_names(getcwd() + "/v2/fails/")
    return __generate_test_suite(
        2,
        tests,
        fails,
    )

def generate_test_suite_v3():
    """wrapper for generate_test_suite for v3"""
    tests = __get_file_names(getcwd() + "/v3/tests/")
    fails = __get_file_names(getcwd() + "/v3/fails/")
    return __generate_test_suite(
        3,
        tests,
        fails,
    )

def generate_test_suite_v4():
    """wrapper for generate_test_suite for v4"""
    tests = __get_file_names(getcwd() + "/v4/tests/")
    fails = __get_file_names(getcwd() + "/v4/fails/")
    return __generate_test_suite(
        4,
        tests,
        fails,
    )

async def main():
    """main entrypoint: argparses, delegates to test scaffold, suite generator, gradescope output"""
    if not sys.argv:
        raise ValueError("Error: Missing version number argument")
    version = sys.argv[1]
    zero_credit = len(sys.argv) > 2 and sys.argv[2] == '--zero-credit'
    is_prod = environ.get("PROD", False)
    module_name = f"interpreterv{version}"
    try:
        interpreter = importlib.import_module(module_name)
        print(f"Module '{module_name}' imported successfully.")
    except SyntaxError as e:
        print(f"Syntax error in module '{module_name}': {e}")
        write_gradescope_output_failure("Syntax error in submission. Test cases cannot run.", is_prod)
        return
    except ModuleNotFoundError as e:
        print(f"ModuleNotFound error in module '{module_name}': {e}")
        write_gradescope_output_failure(f"MissingModuleError ({e}) when loading submission. Possibly caused by"
                                         "missing code file or incorrect file naming. Test cases cannot run.", is_prod)
        return
    except Exception as e:
        print(f"Exception in module '{module_name}': {e}")
        write_gradescope_output_failure("Failed to load submission module. Test cases cannot run.", is_prod)
        return

    scaffold = TestScaffold(interpreter)

    match version:
        case "1":
            tests = generate_test_suite_v1()
        case "2":
            tests = generate_test_suite_v2()
        case "3":
            tests = generate_test_suite_v3()
        case "4":
            tests = generate_test_suite_v4()    
        case _:
            raise ValueError("Unsupported version; expect one of {1, 2, 3, 4}")

    results = await run_all_tests(scaffold, tests, zero_credit=zero_credit)
    total_score = get_score(results) / len(results) * 100.0
    print(f"Total Score: {total_score:9.2f}%")

    # flag that toggles write path for results.json
    write_gradescope_output(results, is_prod)


if __name__ == "__main__":
    asyncio.run(main())