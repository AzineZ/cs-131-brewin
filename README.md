# CS 131 Fall 2024 - Project Autograder

Hi there! This is a repo that contains an open-source subset of the autograder we'll be using for [CS 131 - Fall 2024](https://ucla-cs-131.github.io/fall-24-website/)'s course-long project: making an interpreter.

Using this repository / testing locally is **entirely optional**. It does not directly affect your grade. **You are free to only submit to Gradescope!**

This repository contains:

- the **full source code** for the autograder we deploy to Gradescope
- 20% of the test cases we evaluate your code on; these are the test cases that are public on Gradescope
    - each version of the project is in a `v*` folder;
    - the `tests` subdirectory contains source (`.br`) files for programs that should interpret and run without errors
    - the `fails` subdirectory contains source (`.br`) files for programs that should interpret successfully, but error

This repository does not contain:

- 80% of the test cases we evaluate your code on
- the plagiarism checker, which is closed-source
- the Docker configuration for the deployment; this is managed by Gradescope.
- canonical solutions for the past projects - those are in the [project template repo](https://github.com/UCLA-CS-131/fall-24-project-starter)

We'll note that with the current setup, we grant **five seconds for each test case to run**.

We've made a [separate repository for project template code](https://github.com/UCLA-CS-131/fall-24-project-starter).

## Usage

### Setup

1. **Make sure you're using Python 3.11**
2. Clone this repo and navigate to its root directory

Now, you're ready to test locally.

### Testing Locally

To test locally, you will additionally need a **working implementation** of the project version you're trying to test (your interpreter file and any additional files that you created that it relies on)

Place this in the same directory as `tester.py`. Then, to test project 1 for example,

```sh
python tester.py 1
```

```sh
Running 6 tests...
Running v1/tests/test_add1.br...  PASSED
Running v1/tests/test_print_const.br...  PASSED
Running v1/tests/test_print_var.br...  PASSED
Running v1/fails/test_bad_var1.br...  PASSED
Running v1/fails/test_invalid_operands1.br...  PASSED
Running v1/fails/test_unknown_func_call.br...  PASSED
6/6 tests passed.
Total Score:    100.00%
```

Note: we also output the results of the terminal output to `results.json`.

## Bug Bounty

If you're a student and you've found a bug - please let the TAs know (confidentially)! If you're able to provide a minimum-reproducible example, we'll buy you a coffee - if not more!

## Licensing and Attribution

This code is distributed under the [MIT License](https://github.com/UCLA-CS-131/fall-23-autograder/blob/main/LICENSE).

Have you used this code? We'd love to hear from you! [Submit an issue](https://github.com/UCLA-CS-131/fall-24-autograder/issues) or send us an email ([dboyan@cs.ucla.edu](mailto:dboyan@cs.ucla.edu)).
