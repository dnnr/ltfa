"""
Export the consolidated transactions as a beancount ledger, for use with fava.

Every ltfa transaction becomes a beancount entry with exactly two postings: the
account it happened on and a counter account derived from what ltfa already
knows about it (neutral inter-account transfer, salary, capital gain, or plain
spending/income). The counter posting is the exact negation of the first, so
every entry balances by construction.
"""

import logging
import re
import unicodedata
from typing import Any, Iterator, cast

import pandas as pd

from ltfa.util import LtfaError

CURRENCY = 'EUR'

# Asset accounts get an asset-type tier so that fava's account tree groups them
# in a useful way.
ASSET_TIERS = {
    'liquidity': 'Liquidity',
    'shared-liquidity': 'Shared',
    'investment': 'Investment',
    'misc': 'Misc',
}

COUNTER_ACCOUNTS = [
    'Assets:Transfers',
    'Equity:Opening-Balances',
    'Expenses:Uncategorized',
    'Income:CapitalGains',
    'Income:Salary',
    'Income:Uncategorized',
]


def sanitize(name: str) -> str:
    """Turn an arbitrary ltfa account name into a valid beancount name
    component (letters, digits and dashes, starting with a letter or digit)."""
    # ponytail: no collision handling, two account names differing only in
    # stripped characters would merge. Disambiguate if that ever happens.
    ascii_name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()
    component = re.sub(r'[^A-Za-z0-9-]', '', ascii_name)
    if not component or not component[0].isalnum():
        component = 'X' + component
    return component[0].upper() + component[1:]


def asset_account(account: str, asset_type: str) -> str:
    return 'Assets:{}:{}'.format(ASSET_TIERS.get(asset_type, 'Misc'), sanitize(account))


def counter_account(txn) -> str:
    """Pick the counter posting's account. First match wins."""
    if txn.peeraccount == 'ltfa':
        # The synthetic opening transaction added by accounts_to_dataframes()
        return 'Equity:Opening-Balances'
    if txn.isneutral:
        # Both legs of a transfer route through the clearing account, so it
        # nets to zero. A non-zero balance flags an unpaired neutral txn.
        return 'Assets:Transfers'
    if txn.salary:
        return 'Income:Salary'
    if txn.asset_type == 'investment':
        return 'Income:CapitalGains'
    return 'Income:Uncategorized' if txn.value > 0 else 'Expenses:Uncategorized'


def describe(txn) -> str:
    """Render the payee/narration part. Beancount reads a lone string as the
    narration, which is what we want when there is no peer name."""
    def quote(value) -> str:
        text = value if isinstance(value, str) else ''
        return '"{}"'.format(text.replace('"', "'"))

    payee = quote(txn.peername)
    return payee + ' ' + quote(txn.subject) if payee != '""' else quote(txn.subject)


def rows(txns: pd.DataFrame) -> Iterator[Any]:
    """itertuples() with the row type widened. pandas-stubs types every cell as
    a union of all possible dtypes, which nothing downstream can work with."""
    return cast(Iterator[Any], txns.itertuples())


def make(txns: pd.DataFrame, file) -> None:
    def p(s=''):
        print(s, file=file)

    txns = txns[txns.value != 0]
    if txns.empty:
        raise LtfaError('Got no non-zero transactions to export')

    opening_date = txns.index.min().date()
    accounts = sorted({asset_account(t.account, t.asset_type) for t in rows(txns)})

    p('option "title" "ltfa"')
    p('option "operating_currency" "{}"'.format(CURRENCY))
    p()
    for account in accounts + COUNTER_ACCOUNTS:
        p('{} open {} {}'.format(opening_date, account, CURRENCY))

    for txn in rows(txns):
        p()
        p('{} * {}'.format(txn.Index.date(), describe(txn)))
        for account, value in ((asset_account(txn.account, txn.asset_type), txn.value),
                               (counter_account(txn), -txn.value)):
            p('  {:<40} {:>12.2f} {}'.format(account, value, CURRENCY))

    logging.debug('Exported {} transactions on {} accounts as beancount'.format(len(txns), len(accounts)))
