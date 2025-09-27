import logging
import sys
import yaml
import dateutil.parser
import dateutil.relativedelta
import datetime
import argparse
import os
import appdirs
from pathlib import Path
import itertools
import pandas as pd
from types import SimpleNamespace
from typing import Any

from ltfa.account import Account, Transaction
from ltfa import plotting_bokeh
from ltfa.analysis import Analysis
from ltfa.util import LtfaError
from ltfa.util import file_or_stdout

def run(args) -> None:
    # Set up logging
    log_level = args.loglevel
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level, stream=sys.stdout)
    accounts: list[Account] = []

    # Read and parse configuration
    try:
        config = yaml.load(args.config, Loader=yaml.CSafeLoader)

        for accountcfg in config['accounts']:
            if not accountcfg.get('active'):
                logging.info("Ignoring inactive account: " + accountcfg['name'])
                continue

            account = Account(accountcfg)
            accounts.append(account)

        annotations = {category: {dateutil.parser.parse(date).date(): text for date, text in entries.items()} for category, entries in config.get('annotations', {}).items()}
        salary_matchers = config.get('salary-matchers', [])
    except yaml.YAMLError as e:
        logging.error('Failed to parse configuration file: ' + str(e))
        raise e

    if not accounts:
        raise LtfaError('No valid account configurations')

    # Stage 1 (load everything that is independent from other accounts)
    for a in accounts:
        a.stage1()

    # Stage 2 (load everything that needs info from other accounts)
    for a in accounts:
        a.stage2(accounts)

    # Stage 3 (finalization based on now-complete knowledge)
    for a in accounts:
        a.stage3()

    for a in accounts:
        if a.txns:
            logging.info(
                "{}: Final balance after {} txns: {}, € {}".format(
                    a.name, len(a.txns), a.txns[-1].date, a.txns[-1].balance
                )
            )
        else:
            logging.warning("{}: Got no transactions at all".format(a.name))

    accounts_df = accounts_to_dataframes(accounts)
    analysis = Analysis(accounts_df, salary_matchers)

    if args.investment_report and analysis.has_capgains:
        with file_or_stdout(args.investment_report) as fh:
            analysis.make_capgains_analysis(fh)

    if args.monthly_overview:
        if not args.month_for_overview:
            one_month_ago = datetime.date.today() - dateutil.relativedelta.relativedelta(months=1)
            args.month_for_overview = one_month_ago.strftime('%Y-%m')
        with file_or_stdout(args.monthly_overview) as fh:
            analysis.make_monthly_overview(fh, args.month_for_overview)

    if args.bokeh:
        plotting_bokeh.make(accounts_df, annotations, analysis, args.bokeh)


def parse_args(args) -> argparse.Namespace:
    argparser = argparse.ArgumentParser(description="")

    argparser.add_argument(
        '--config',
        metavar='FILE',
        type=argparse.FileType('r'),
        dest='config',
        help="Configuration file to use (default: %(default)s ",
        default=os.path.join(appdirs.user_config_dir('ltfa'), 'ltfa.yaml'),
    )

    argparser.add_argument(
        '-d', '--debug',
        help="Enable debug messages",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.WARNING,
    )

    argparser.add_argument(
        '-v', '--verbose',
        help="Be verbose",
        action="store_const", dest="loglevel", const=logging.INFO,
    )

    argparser.add_argument('-B', '--bokeh', type=Path, metavar='FILE', help='Write bokeh visualization to this file')
    argparser.add_argument('-I', '--investment-report', nargs='?', type=Path, metavar='FILE', const='/dev/stdout', help='File to write investment report into (default: stdout)')

    argparser.add_argument('-M', '--monthly-overview', nargs='?', type=Path, metavar='FILE', const='/dev/stdout', help='File to write montly report into (default: stdout)')
    argparser.add_argument('--month-for-overview', nargs='?', type=str, metavar='YYYY-MM', help='Month to use for monthly overview (default: last month)')

    return argparser.parse_args(args)


def accounts_to_dataframes(accounts) -> list[pd.DataFrame]:
    """ Turn accounts into dataframes as a preparation for stacking them. """
    # First determine the time bounds of all data, so that every dataframe can
    # cover the full range
    alldates = [t.date for t in itertools.chain(*[a.txns for a in accounts])]
    beginningoftime = min(alldates)
    endoftime = max(alldates)

    ret = []
    for account in accounts:
        asset_type = account.config.get('asset-type')
        # Assume that the initial balance has always been there:
        startat: list[tuple] = [(
            beginningoftime,  # date
            float(account.initial_balance),  # value
            'ltfa',  # peername
            'ltfa',  # peeraccount
            'Initial account balance',  # subject
            True,  # isneutral
            account.name,  # account
            asset_type,  # asset_type
        )]

        # Add an empty transaction at the end
        endat: list[tuple] = [(endoftime, float(0), [])]

        # Turn account transactions into dataframe
        txns = pd.DataFrame(
            startat + [(
                txn.date,
                float(txn.value),
                getattr(txn, 'peername', 'n/a'),
                txn.peeraccount,
                txn.subject,
                txn.isneutral,
                account.name,
                asset_type,
            ) for txn in account.txns] + endat,
            columns=[
                'date',
                'value',
                'peername',
                'peeraccount',
                'subject',
                'isneutral',
                'account',
                'asset_type',
            ]).astype({
                'isneutral': 'boolean',
            })

        # Converting all dates in a single Series call is much faster than
        # doing it within he list comprehension:
        txns['date'] = pd.to_datetime(txns.date, format='%Y-%m-%d')

        txns.set_index('date', inplace=True)

        # Sum up transactions of each day
        dailies = txns.groupby(level='date').agg(dict(value="sum"))

        # Compute balance for each day that has a transaction
        dailies['balance'] = dailies.value.cumsum()
        # Round all balances to get rid of accumulating floating point errors introduced by cumsum()
        dailies.balance = dailies.balance.round(decimals=5)

        # Drop the helper dummy txn at the end because it was only relevant for
        # balance computation. Do keep, however, the one at the beginning, so
        # that any subsequent cumsum() runs on the transactions produce the
        # correct results.
        txns = txns[:-1]

        ret.append(SimpleNamespace(meta=account, dailies=dailies, txns=txns))

    # Just for debugging output:
    #  total_txns = sum(len(x.dailies) for x in ret)
    #  print(f"Turned {len(ret)} accounts into dataframes, {total_txns} transactions in total")

    return ret
