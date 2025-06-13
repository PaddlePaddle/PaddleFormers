[简体中文🀄](../CONTRIBUTING.md) |  **English**🌎

# Contributing to PaddleFormers

We highly welcome and value your contributions to `PaddleFormers`. The first step to start your contribution is to sign the [PaddlePaddle Contributor License Agreement](https://cla-assistant.io/PaddlePaddle/PaddleFormers).

This document explains our workflow and work style:

## Finding out what to work on Workflow

## Development Workflow

PaddleFormers uses the [Git branching model](http://nvie.com/posts/a-successful-git-branching-model/).  The following steps guide usual contributions.

#### 1. Fork

   Our development community has been growing fastly; it doesn't make sense for everyone to write into the official repo.  So, please file Pull Requests from your fork.  To make a fork,  just head over to the GitHub page and click the ["Fork" button](https://help.github.com/articles/fork-a-repo/).

#### 2. Clone

   To make a copy of your fork to your local computers, please run

   ```bash
   git clone https://github.com/<your-github-account>/PaddleFormers
   cd PaddleFormers
   ```

#### 3. Create the local feature branch

   For daily works like adding a new feature or fixing a bug, please open your feature branch before coding:

   ```bash
   git checkout -b my-cool-feature
   ```

#### 4. Set up the development environment

   Before you start coding, you need to setup the development environment. We highly recommend doing all your development in a virtual environment such as
   [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/en/latest/). After you setup and activated your virtual environment,
   run the following command:

   ```bash
   make install
   ```

   This will setup all the dependencies of `PaddleFormers` as well as the [`pre-commit`](http://pre-commit.com/) tool.

   If you are working on the `examples` or `applications` module and require importing from `PaddleFormers`, make sure you install `PaddleFormers` in editable mode.
   If `PaddleFormers` is already installed in the virtual environment, remove it with `pip uninstall PaddleFormers` before reinstalling it in editable mode with
   `pip install -e .`

#### 5. Develop

   As you develop your new exciting feature, keep in mind that it should be covered by unit tests. All of our unit tests can be found under the `tests` directory.
   You can either modify existing unit test to cover the new feature, or create a new test from scratch.
   As you finish up the your code, you should make sure the test suite passes. You can run the tests impacted by your changes like this:

   ```bash
   pytest tests/<test_to_run>.py
   ```

#### 6. Commit

   We utilizes [`pre-commit`](http://pre-commit.com/) (with [black](https://black.readthedocs.io/en/stable/), [isort](https://pycqa.github.io/isort/) and
   [flake8](https://flake8.pycqa.org/en/latest/) under the hood) to check the style of code and documentation in every commit. When you run run `git commit`, you will see
   something like the following:

   ```
    ➜  (my-virtual-env) git commit -m "commiting my cool feature"
    black....................................................................Passed
    isort....................................................................Passed
    flake8...................................................................Passed
    check for merge conflicts................................................Passed
    check for broken symlinks............................(no files to check)Skipped
    detect private key.......................................................Passed
    fix end of files.....................................(no files to check)Skipped
    trim trailing whitespace.............................(no files to check)Skipped
    CRLF end-lines checker...............................(no files to check)Skipped
    CRLF end-lines remover...............................(no files to check)Skipped
    No-tabs checker......................................(no files to check)Skipped
    Tabs remover.........................................(no files to check)Skipped
    copyright_checker........................................................Passed
   ```

   But most of the time things don't go so smoothly. When your code or documentation doesn't meet the standard, the `pre-commit` check will fail.
   ```
    ➜  (my-virtual-env) git commit -m "commiting my cool feature"
    black....................................................................Passed
    isort....................................................................Failed
    - hook id: isort
    - files were modified by this hook

    Fixing examples/information_extraction/waybill_ie/run_ernie_crf.py

    flake8...................................................................Passed
    check for merge conflicts................................................Passed
    check for broken symlinks............................(no files to check)Skipped
    detect private key.......................................................Passed
    fix end of files.....................................(no files to check)Skipped
    trim trailing whitespace.............................(no files to check)Skipped
    CRLF end-lines checker...............................(no files to check)Skipped
    CRLF end-lines remover...............................(no files to check)Skipped
    No-tabs checker......................................(no files to check)Skipped
    Tabs remover.........................................(no files to check)Skipped
    copyright_checker........................................................Passed
   ```

   But **don't panic**!
   Our tooling will fix most of the style errors automatically. Some errors will need to be addressed manually. Fortunately, the error messages are straight forward and
   the errors are usually simple to fix. After addressing the errors, you can run `git add <files>` and `git commit` again, which will trigger `pre-commit` again.
   Once the `pre-commit` checks pass, you are ready to push the code.

   [Google][http://google.com/] or [StackOverflow](https://stackoverflow.com/) are great tools to help you understand the code style errors.
   Don't worry if you still can't figure it out. You can commit with `git commit -m "style error" --no-verify` and we are happy to help you once you create a Pull Request.

#### 7. Keep pulling

   An experienced Git user pulls from the official repo often -- daily or even hourly, so they notice conflicts with others work early, and it's easier to resolve smaller conflicts.

   ```bash
   git remote add upstream https://github.com/PaddlePaddle/PaddleFormers
   git pull upstream develop
   ```

#### 8. Push and file a pull request

   You can "push" your local work into your forked repo:

   ```bash
   git push origin my-cool-stuff
   ```

   The push allows you to create a pull request, requesting owners of this [official repo](https://github.com/PaddlePaddle/PaddleFormers) to pull your change into the official one.

   To create a pull request, please follow [these steps](https://help.github.com/articles/creating-a-pull-request/).

#### 9. Delete local and remote branches

   To keep your local workspace and your fork clean, you might want to remove merged branches:

   ```bash
   git push origin my-cool-stuff
   git checkout develop
   git pull upstream develop
   git branch -d my-cool-stuff
   ```

## Code Review

-  Please feel free to ping your reviewers by @-mentioning the in the Pull Request.  Please do this after your pull request passes the CI.

- Please answer reviewers' every comment.  If you are to follow the comment, please write "Done"; Otherwise, please start a discussion under the comment.

- If you don't want your reviewers to get overwhelmed by email notifications, you might reply their comments by [in a batch](https://help.github.com/articles/reviewing-proposed-changes-in-a-pull-request/).
