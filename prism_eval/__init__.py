"""Evaluation harness shared by PRISM's evaluator and its reports.

The package holds the parts of an evaluation run that are not specific to any
one solver: where instances come from (``instances``), how a measurement is
recorded (``results``), and how measurements are aggregated into comparisons
(``summary``). ``test.py`` supplies PRISM itself; baselines are expected to
grow into their own adapters over the same row schema.
"""
