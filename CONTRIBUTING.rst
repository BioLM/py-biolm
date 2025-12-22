.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/BioLM/py-biolm/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

BioLM AI could always use more documentation, whether as part of the
official BioLM AI docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/BioLM/py-biolm/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `biolmai` for local development.

1. Fork the `biolmai` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/biolmai.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    $ mkvirtualenv biolmai
    $ cd biolmai/
    $ python setup.py develop

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the
   tests, including testing other Python versions with tox::

    $ flake8 biolmai tests
    $ python setup.py test or pytest
    $ tox

   To get flake8 and tox, just pip install them into your virtualenv.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for all supported Python versions. Check
   https://github.com/BioLM/py-biolm/actions
   and make sure that the tests pass for all supported Python versions.

Tips
----

To run a subset of tests::

$ pytest tests.test_biolmai


Commit Message Format
---------------------

This project uses `Conventional Commits <https://www.conventionalcommits.org/>`_ for automatic version bumping. Use the following prefixes in your commit messages:

* ``feat:`` - New feature (minor version bump: 0.2.8 → 0.3.0)
* ``fix:`` - Bug fix (patch version bump: 0.2.8 → 0.2.9)
* ``BREAKING:`` - Breaking change (major version bump: 0.2.8 → 1.0.0)
* ``docs:`` - Documentation only (no version bump)
* ``chore:`` - Maintenance tasks (no version bump)
* ``refactor:`` - Code refactoring (no version bump)

Examples::

    $ git commit -m "feat: add batch processing support"
    $ git commit -m "fix: resolve authentication timeout issue"
    $ git commit -m "BREAKING: change API endpoint structure"
    $ git commit -m "docs: update installation instructions"

Deploying
---------

This project uses automatic semantic versioning and release management.

**For Contributors:**

1. Make your changes and commit using conventional commit format (see above).
2. Push to your branch and create a pull request.
3. After the PR is merged to ``main``, the CI workflow will automatically:
   * Analyze commit messages
   * Bump version if needed (based on commit types)
   * Update version in ``biolmai/__init__.py``, ``setup.py``, and ``pyproject.toml``
   * Update ``HISTORY.rst`` changelog
   * Create a git tag
   * Push the tag to GitHub

**For Maintainers:**

1. After a version tag is created automatically, create a GitHub release:
   * Go to https://github.com/BioLM/py-biolm/releases/new
   * Select the tag created by semantic-release
   * Add release notes (or use auto-generated ones)
   * Click "Publish release"

2. The publish workflow will automatically:
   * Build the package
   * Publish to PyPI (or TestPyPI for release candidates)
   * Update the release with publish status

**Note:** The old ``production`` branch deployment is still available for backward compatibility, but release-based publishing is the preferred method.
