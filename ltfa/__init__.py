import logging
import sys
import yaml
import dateutil.parser
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

def run(args) -> None:
    # Set up logging
    log_level = args.loglevel
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level, stream=sys.stdout)
    accounts: list[Account] = []

    # Read and parse configuration
    try:
        config = yaml.safe_load(args.config)

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

    # Stage 3 (perform some txn classifications based on now-complete knowledge)
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

    accounts = accounts_to_dataframes(accounts)
    analysis = Analysis(accounts, salary_matchers)

    if args.investment_report and analysis.has_capgains:
        with open(args.investment_report, 'w') as fh:
            analysis.make_capgains_analysis(fh)

    if args.bokeh:
        plotting_bokeh.makeplot_balances(accounts, annotations, analysis, args.bokeh)


def parse_args(args) -> argparse.Namespace:
    argparser = argparse.ArgumentParser(description="")

    argparser.add_argument(
        '--config',
        metavar='FILE',
        type=argparse.FileType('r'),
        dest='config',
        help="Configuration file to use (default: %(default)s ",
        default=os.path.join(appdirs.user_config_dir('ltfa'), 'ltfa.conf'),
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
        # Assume that the initial balance has always been there:
        startat: list[tuple] = [(pd.to_datetime(beginningoftime), float(account.initial_balance), 'ltfa', 'Initial account balance', True)]

        # Add an empty transaction at the end
        endat: list[tuple] = [(pd.to_datetime(endoftime), float(0), [])]

        # Turn account transactions into dataframe
        txns = pd.DataFrame(
            startat + [(pd.to_datetime(txn.date), float(txn.value), getattr(txn, 'peername', 'n/a'), txn.subject, txn.isneutral) for txn in account.txns] + endat,
            columns=['date', 'value', 'peername', 'subject', 'isneutral']).astype(
                {
                    'isneutral': 'boolean',
                }
            )
        txns.set_index('date', inplace=True)

        # Sum up transactions of each day
        dailies = txns.groupby(level='date').agg(dict(value=sum))

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
