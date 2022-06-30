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
        'simple_checking',
        'simple_savings',
        'simple_savings_monthly',
        'savings_varying_interest',
        'savings_negative_gains',
     ),
)
def test_real_data_snapshot(tmpdir, request, scenario):
    scenario_dir = Path(__file__).parent / 'scenarios' / scenario
    assert scenario_dir.is_dir()

    with open(scenario_dir / f'{scenario}.log', "r") as logfh:
        expected_log = logfh.readlines()

    # Transform log into format that matches LogCapture:
    expected_log = [("root", *l.rstrip().split(": ", 1)) for l in expected_log]

    args = ltfa.parse_args([
        "--config", str(scenario_dir / f'{scenario}.conf'),
        '--output-dir', str(tmpdir),
    ])

    with testfixtures.LogCapture(level=logging.INFO) as log_capture:
        ltfa.run(args)
        log_capture.check(*expected_log)

    balances_html = tmpdir / 'ltfa_bokeh.html'
    assert balances_html.exists()

    #  capgains_html = tmpdir / 'ltfa_capgains_mpld3.html'
    #  assert capgains_html.exists()
