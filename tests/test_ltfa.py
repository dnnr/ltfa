import logging
import testfixtures
import pytest
from pathlib import Path
import os

import ltfa
import ltfa.util


@pytest.mark.parametrize(
    'scenario',
    (
        'savings_negative_gains',
        'savings_varying_interest',
        'shared_ownership',
        'simple_checking',
        'simple_savings',
        'simple_savings_monthly',
     ),
)
def test_real_data_snapshot(tmpdir, request, scenario):
    scenario_dir = Path(__file__).parent / 'scenarios' / scenario
    assert scenario_dir.is_dir()

    with open(scenario_dir / f'{scenario}.log', "r") as fh:
        # Transform log into format that matches LogCapture:
        expected_log = [("root", *l.rstrip().split(": ", 1)) for l in fh.readlines()]

    expected_investment_report = scenario_dir / f'{scenario}.investment_report.txt'

    bokeh_html = tmpdir / 'bokeh.html'
    assert not bokeh_html.exists()

    investment_report = tmpdir / 'investment_report.txt'
    assert not investment_report.exists()

    args = ltfa.parse_args([
        '--config', str(scenario_dir / f'{scenario}.conf'),
        '--bokeh', str(bokeh_html),
        '-I', str(investment_report),
    ])

    with testfixtures.LogCapture(level=logging.INFO) as log_capture:
        ltfa.run(args)
        log_capture.check(*expected_log)

    assert bokeh_html.exists()

    if expected_investment_report.exists():
        with open(scenario_dir / f'{scenario}.investment_report.txt', "r") as left:
            with open(investment_report, 'r') as right:
                assert left.readlines() == right.readlines()
    else:
        assert not investment_report.exists()
