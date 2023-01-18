import csv
import datetime
import logging

from decimal import Decimal
from operator import attrgetter

from ltfa.transaction import Transaction
from ltfa.util import LtfaError


class YamlTxnLoader:
    """ Just a container for static YAML txn parsing function(s) """

    @staticmethod
    def parse_yaml_transactions(txns):
        for rawtxn in txns:
            txnargs = {
                'value': Decimal(rawtxn['value']),
                'date': rawtxn['date'],
                'subject': rawtxn.get('subject') or '',
                'isneutral': rawtxn.get('isneutral') or None,
            }
            for k in ('subject', 'peername', 'peeraccount', 'peerbic'):
                txnargs[k] = rawtxn.get(k) or ''

            spurious = rawtxn.keys() - txnargs.keys()
            if spurious:
                raise LtfaError("parse_manual_transactions(): Unknown field name(s): {}".format(', '.join(list(spurious))))
            yield Transaction(**txnargs)


class CsvLoader:
    """ Just a container for static CSV parsing functions """

    @staticmethod
    def make_decimal(s, formatcfg):
        if 'thousands-sep' in formatcfg:
            s = s.replace(formatcfg['thousands-sep'], '')
        if 'decimal-mark' in formatcfg:
            s = s.replace(formatcfg['decimal-mark'], '.')

        return Decimal(s)

    @staticmethod
    def load_txns(filepath, formatcfg, filterlist):
        with open(filepath, 'r', errors='replace') as csvfile:
            delimiter = formatcfg.get('delimiter') or None

            dialect = 'excel'
            has_header = False
            if delimiter != ';':
                # The sniffer has trouble with semicolons
                dialect = csv.Sniffer().sniff(csvfile.read(1024), delimiters=delimiter)
                csvfile.seek(0)

                has_header = csv.Sniffer().has_header(csvfile.read(1024))
                csvfile.seek(0)

            if has_header:
                header = list(csv.reader(csvfile, dialect))[0]
                csvfile.seek(0)
                colmap = dict(reversed(x) for x in enumerate(header))
            else:

                class IdentDict(dict):
                    __missing__ = lambda self, key: key

                colmap = IdentDict()

            if delimiter is not None:
                csvreader = csv.reader(csvfile, delimiter=delimiter)
            else:
                csvreader = csv.reader(csvfile, dialect=dialect)

            txns = []
            for row in list(csvreader)[1 if has_header else 0:]:
                # Convert row to field map according to specified column indices.
                fieldmap = dict()
                for key in formatcfg['columns'].keys():
                    colidx = colmap[formatcfg['columns'][key]]
                    fieldmap[key] = row[colidx]

                # Parse date field (assumed mandatory)
                fieldmap['date'] = datetime.datetime.strptime(
                        fieldmap['date'],
                        formatcfg['dateformat']
                        ).date()

                # Parse value field (assumed mandatory)
                fieldmap['value'] = CsvLoader.make_decimal(fieldmap['value'], formatcfg)

                # Parse balance field (if present)
                if 'balance' in fieldmap:
                    fieldmap['balance'] = CsvLoader.make_decimal(fieldmap['balance'], formatcfg)

                if fieldmap['value'] == 0 and not 'balance' in fieldmap:
                    logging.debug("Ignoring zero-value CSV entry with no balance: {}".format(fieldmap))
                    continue

                # Verify transaction against configured filters:
                if not all(fieldmap[fkey] == fval for fkey, fval in filterlist.items()):
                    logging.debug("Ignoring entry not matching filter conditions: {}".format(fieldmap))
                    continue

                # Construct Transaction object using values expected by the
                # constructor. Missing values are assumed to be optional and
                # default-initialized by the constructor.
                txn_fields = [
                    'date',
                    'value',
                    'subject',
                    'peername',
                    'peeraccount',
                    'peerbic',
                    'balance',
                    'account',
                    ]
                txn = Transaction(**{k: fieldmap[k] for k in txn_fields if k in fieldmap})
                txns.append(txn)

            txns.sort(key=attrgetter('date'))

            return txns
