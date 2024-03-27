import logging
import testfixtures
import pytest
from pathlib import Path
import os
import shutil

import ltfa
import ltfa.util


@pytest.mark.parametrize(
    'scenario',
    (
        'first_txn_is_gain',
        'savings_negative_gains',
        'savings_varying_interest',
        'shared_ownership',
        'simple_checking',
        'simple_savings_100p',
        'simple_savings_5p_payout_dec31',
        'simple_savings_5p_payout_feb28',
        'simple_savings_5p_payout_jan1',
        'simple_savings_5p_payout_jan2',
        'simple_savings_5p_payout_jan2_noleapyears',
        'simple_savings_5p_payout_march31',
        'simple_savings_5p_payout_march31_noleapyears',
        'simple_savings_5p_with_noise',
        'simple_savings_monthly',
     ),
)
def test_real_data_snapshot(tmpdir, request, scenario):
    scenario_dir = Path(__file__).parent / 'scenarios' / scenario
    assert scenario_dir.is_dir()

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

        # Uncomment to rewrite log snapshot:
        #  with open(scenario_dir / f'{scenario}.log', "w") as fh:
            #  for line in log_capture:
                #  print(': '.join(line[1:]), file=fh)

        with open(scenario_dir / f'{scenario}.log', "r") as fh:
            # Transform log into format that matches LogCapture:
            expected_log = [("root", *l.rstrip().split(": ", 1)) for l in fh.readlines()]

        log_capture.check(*expected_log)

    assert bokeh_html.exists()

    #  Uncomment to rewrite report snapshot:
    #  if investment_report.exists():
        #  shutil.copyfile(investment_report, scenario_dir / f'{scenario}.investment_report.txt', )

    if expected_investment_report.exists():
        with open(scenario_dir / f'{scenario}.investment_report.txt', "r") as left:
            with open(investment_report, 'r') as right:
                assert left.readlines() == right.readlines()
    else:
        assert not investment_report.exists()
