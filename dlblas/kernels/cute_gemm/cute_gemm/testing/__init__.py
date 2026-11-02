# Copyright (c) 2026, Tri Dao.
"""Reusable pytest plugin for cute_gemm test workflows.

This subpackage's only contents is the reusable pytest plugin in
:mod:`cute_gemm.testing.pytest_plugin`, which wires the ``--async-compile`` pool
(defer-and-retry kernel compilation) into a pytest run::

    # In a downstream project's conftest.py:
    pytest_plugins = ["cute_gemm.testing.pytest_plugin"]
"""
