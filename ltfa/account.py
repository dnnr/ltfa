import dateutil.parser
import glob
import logging
import math
import os
import simplejson

from collections import deque, defaultdict
from decimal import Decimal
from operator import attrgetter
from operator import itemgetter
from types import SimpleNamespace
from typing import Optional

from ltfa.loaders import YamlTxnLoader, CsvLoader
from ltfa.transaction import Transaction
from ltfa.util import formatifnonempty
from ltfa.util import LtfaError


class Account:
    def __init__(self, config) -> None:
        self.name = config['name']
        self.config = config
        self.txns: list[Transaction] = []
        self.initial_balance: Optional[Decimal] = None

    def __repr__(self) -> str:
        return f"<Account at 0x{id(self):x}, name={self.name}, {len(self.txns)} txns>"

    def stage1(self) -> None:
        if 'asset-type' not in self.config:
            raise LtfaError(f'No asset-type defined for account "{self.name}"')

        if self.config['asset-type'] not in ['liquidity', 'investment', 'misc', 'shared-liquidity']:
            raise LtfaError(f'Invalid asset-type set for account "{self.name}": {self.config["asset-type"]}')

        if self.config['asset-type'] == 'shared-liquidity':
            if not 'share-owned' in self.config:
                raise LtfaError(f'Missing share-owned value for shared-liquidity account "{self.name}"')
            if self.config.get('autoinfer-from') or []:
                raise LtfaError(f'Setting any autoinfer-from matchers for a shared-liquidity account is not allowed: {self.name}')
        else:
            if 'share-owned' in self.config:
                raise LtfaError(f'{self.name}: Setting share-owned is only allowed for for shared-liquidity accounts')


        self._load_csv_txns()
        self._load_static_txns()

        self._recompute_balances()

    def stage2(self, accounts) -> None:
        self._autoinfer_txns(accounts)

        # Reduce all transaction values and balances according to shared ownership
        self._apply_shared_ownership()

        # Balance checkpoint insertion relies on a correct initial_balance, so
        # recompute beforehand:
        self._recompute_balances()

        # This must always be the very last step because it relies on the txn
        # balances being accurate:
        self._insert_balance_checkpoints()

        # Auto-infer again to handle possible inter-account balance checkpoints
        self._autoinfer_txns(accounts)

        self._recompute_balances()
        for t in self.txns:
            if (t.value != 0):  # A little less noise
                logging.debug(f'{self.name}: Final recomputed balance on {t.date}: {t.balance} after {t}')

        stillnegative = False
        for t in self.txns:
            if t.balance < 0 and not stillnegative:
                logging.warning(f'{self.name}: Negative balance of {t.balance} after: {t}')
            stillnegative = t.balance < 0

    def stage3(self) -> None:
        for txn in self.txns:
            # Any txn still with isneutral==None can now be safely assumed non-neutral.
            if txn.isneutral == None:
                txn.isneutral = False

        self._validate_balances()

    def _recompute_balances(self) -> None:
        share_owned = Decimal(self.config.get('share-owned') or 1)
        balance_start = self.config.get('balance_start') or 0

        if balance_start == 'countback':
            # Infer initial balance from desired final balance and the sum of
            # transaction values:
            balance_end = share_owned * Decimal(self.config.get('balance_end') or 0)
            sumofvalues = sum(t.value for t in self.txns) or Decimal(0)
            balance_start = Decimal(balance_end) - sumofvalues
        else:
            # Use it as is
            balance_start = share_owned * Decimal(balance_start)

        if self.initial_balance is None:
            logging.debug("{}: Setting initial balance: {}".format(self.name, balance_start))
        elif self.initial_balance != balance_start:
            logging.debug(
                "{}: Updating initial balance: {} (was {})".format(self.name, balance_start, self.initial_balance)
            )

        self.initial_balance = balance_start

        i = balance_start
        for t in self.txns:
            i += t.value
            t.balance = i

    def _validate_balances(self) -> None:
        # Since we don't retain the original source order of transactions, the
        # balance might be off for some transactions if there are multiple ones
        # on the same day. It should be correct for at least one of the
        # transactions on the same day, so that's what we check here.
        tmp_txns_per_day = defaultdict(list)
        for txn in self.txns:
            tmp_txns_per_day[txn.date].append(txn)

        any_errors = False
        for date, txns in tmp_txns_per_day.items():
            final_computed_balance = txns[-1].balance
            verification_balances = set(t.balance_only_for_verification for t in txns)

            day_has_verification_balances = any(x != None for x in verification_balances)
            at_least_one_balance_is_correct = final_computed_balance in verification_balances

            if day_has_verification_balances and not at_least_one_balance_is_correct:
                any_errors = True
                logging.error(f'{self.name}: {date}: Final computed balance ({final_computed_balance}) does not match any of the ground truth values ({verification_balances}) of {len(txns)} txns on that day. Transactions listed below:')
                for txn in txns:
                    logging.error(f'{self.name}: Transaction on day ({txn.date}) with balance verification mismatch: {txn}')

        if any_errors:
            raise LtfaError(f'{self.name}: Balance verification has failed (see logged errors)')


    def _insert_txns(self, txns) -> None:
        self.txns.extend(txns)
        # Sort by date, but use inverse value as secondary key so that intraday
        # balances don't end up negative:
        self.txns.sort(key=lambda x: (x.date, -x.value))

    def _load_static_txns(self) -> None:
        txns = self.config.get('transactions') or []
        txns = list(YamlTxnLoader.parse_yaml_transactions(txns))
        txns.sort(key=attrgetter('date'))
        # Note that there are (probably) no balances in those txns yet!

        for t in txns:
            logging.debug("{}: Adding static txn: {}".format(self.name, t.shortstr()))

        self._insert_txns(txns)

    def _autoinfer_txns(self, accounts) -> None:
        matchsets = self.config.get('autoinfer-from') or []

        if not matchsets:
            return

        if self.config['asset-type'] == 'shared-liquidity':
            return

        # Apply defaults to matchsets:
        inferdefaults = self.config.get('autoinfer-from-defaults') or {}
        for matchset in matchsets:
            for defkey, defval in inferdefaults.items():
                if not defkey in matchset:
                    matchset[defkey] = defval

        # Do some necessary type casts:
        for matchset in matchsets:
            for k, v in matchset.items():
                if k == 'value':
                    matchset[k] = Decimal(v)

        # Look for matching transactions in other accounts:
        matches = []
        for account in accounts:
            if account is self or account.config['asset-type'] == 'shared-liquidity':
                continue
            for txn in account.txns:
                if type(txn.peeraccount) is Account:
                    # already auto-inferred, ignore
                    continue
                for matchset in matchsets:
                    ismatch = all(txn.match(*pair) for pair in matchset.items())
                    if ismatch:
                        matches.append((txn, account))
                        # Stop after first matching set:
                        break

        # Construct counter transactions:
        newtxns = []
        for txn, account in matches:
            if txn.isneutral != None:
                logging.warning(
                    "{}: Auto-infer match found in account {} already marked as isneutral={}, this should never happen! Not creating a counter transaction for this.".format(
                        self.name, account.name, txn.isneutral
                    )
                )
                continue

            newtxns.append(
                Transaction(
                    value=-txn.value,
                    # balance will be filled later
                    date=txn.date,
                    subject='AutoInfer: ' + txn.subject,
                    peername=account.name,
                    peeraccount=account,
                    peerbic='',
                    isneutral=True,
                )
            )

            txn.isneutral = True
            # Update peeraccount to point to actual Account object instead
            # of its name:
            txn.peeraccount = self

        # Log new txns in date-order for clarity:
        for t in sorted(newtxns, key=attrgetter('date')):
            logging.debug("{}: Adding auto-inferred txn: {} (via {})".format(self.name, t.shortstr(), t.peername))

        self._insert_txns(newtxns)

    def _apply_shared_ownership(self) -> None:
        if self.config['asset-type'] != 'shared-liquidity':
            return
        share = Decimal(self.config['share-owned'])
        logging.debug(f'{self.name}: Applying shared ownership: {share}')

        for txn in self.txns:
            txn.value *= share
            txn.balance *= share
            if txn.balance_only_for_verification != None:
                txn.balance_only_for_verification *= share

    def _load_csv_txns(self) -> None:
        defs = self.config.get('from-csv') or []
        if type(defs) is dict:
            defs = [defs]

        for c in defs:
            filepath = os.path.expanduser(c['file'])
            logging.debug(f'{self.name}: Loading transactions...')
            txns = CsvLoader.load_txns(filepath, c['format'], c.get('filters') or [])

            # Those are usually a lot, don't log them individually:
            logging.debug("{}: Adding {} txns from CSV".format(self.name, len(txns)))

            self._insert_txns(txns)

    @staticmethod
    def _txndateslice(txns, fromdate, todate=None) -> list[Transaction]:
        """ Helper to select a subset of a transaction list based on
        date limits. The one, last transaction to come before the
        given lower limit is always included, too. """

        txns = sorted(txns, key=attrgetter('date'))
        justbefore = [i for i, t in enumerate(txns) if t.date < fromdate]
        lastjustbefore = justbefore[-1] if justbefore else 0
        txns = txns[lastjustbefore:]
        if todate:
            txns = [t for t in txns if t.date <= todate]
        return txns

    def _insert_balance_interpolation_points(self, balances) -> None:
        # Use first transaction as stand-in for previous checkpoint
        prevcheckpoint = self.txns[0] if self.txns else None

        for newbalance in balances:
            if prevcheckpoint and newbalance.interpolate != 0:
                # Create interpolation points between previous checkpoint and
                # this one, based on the difference from the existing balance
                # value (not the previous checkpoint!).
                # Note that interpolation points assign a value, not a balance!

                daterange = (prevcheckpoint.date, newbalance.date)

                # Determine currently known balance at checkpoint date.
                # (Include prevcheckpoint in the search, so that a match is
                # guaranteed; and it might even be the best fit!)
                lastknown = [t for t in
                             sorted(self.txns + [prevcheckpoint], key=attrgetter('date'))
                             if t.date <= daterange[1]][-1]
                diff2distribute = newbalance.balance - lastknown.balance

                numinterpoints = newbalance.interpolate
                if numinterpoints == -1:
                    # Choose number of interpolation points heuristically:
                    # * around one per month:
                    days = 30
                    # * few enough so each has a minimum amount:
                    minabsamount = 150
                    # * at least two:
                    minnum = 2

                    numinterpoints = (daterange[1] - daterange[0]).days / days

                    avgabsamount = abs(float(diff2distribute)) / numinterpoints
                    if avgabsamount < minabsamount:
                        numinterpoints *= avgabsamount / minabsamount

                    numinterpoints = max(minnum, numinterpoints)
                    numinterpoints = math.ceil(numinterpoints)
                    logging.debug("{}: Automagic numinterpoints: {}".format(self.name, numinterpoints))

                # Build list of dates for interpolation points. The exact
                # date of the balance checkpoint is part of that on purpose, so
                # that we don't end up with surprise negative balances after
                # adding the actual checkpoint transactions later (those should
                # amount to zero then).
                interdates = []
                for i in range(numinterpoints):
                    date = daterange[0] + ((i + 1) * (daterange[1] - daterange[0]) / numinterpoints)
                    interdates.append(date)

                # Collect list of existing balances, in order to keep
                # interpolation points from going into the red.
                # Note that (contrary to previous assumptions), this list must
                # contain *all* balances in within the time window, not just
                # those below a certain limit, since we also need to know when
                # there's enough balance available.
                ballimits = self._txndateslice(self.txns, interdates[0], interdates[-1])

                interpol_txns = []
                for checkp_i, nextinter_date in enumerate(interdates):
                    nextinter_val = diff2distribute / (numinterpoints - checkp_i)

                    # Trim ballimits according to the date for the current
                    # interpolation point:
                    ballimits = self._txndateslice(ballimits, nextinter_date)

                    if ballimits:
                        # Cap interpolation value based on the lowest balance
                        # yet to come:
                        lowestbalance = min(t.balance for t in ballimits)
                        diffdistributed = sum(t.value for t in interpol_txns)
                        minnextval = -(diffdistributed + lowestbalance)

                        if nextinter_val < minnextval:
                            nextinter_val = minnextval

                    # Make nice
                    nextinter_val = round(nextinter_val, 2)

                    t = Transaction(
                            date=nextinter_date,
                            value=nextinter_val,
                            subject='Interpolated balance checkpoint ({}/{})'.format(checkp_i + 1, numinterpoints),
                            # Assuming that balance checkpoints already reflect "hidden" changes in wealth:
                            isneutral=False,
                            )
                    interpol_txns.append(t)
                    logging.debug("{}: Adding balance interpolation point: {}, value={}".format(self.name, t.date, t.value))

                    diff2distribute -= nextinter_val

                # Insert immediately, so that changes already apply in next
                # loop iteration!
                self._insert_txns(interpol_txns)
                self._recompute_balances()

            prevcheckpoint = newbalance

    def _collect_balance_checkpoints(self) -> list[dict]:
        """
        Helper to combine YAML-specified and JSON-loaded balance checkpoints
        into a unified list.
        """
        balances: list[dict] = []
        seen_dates: set = set()

        # Add YAML balances:
        balances.extend(self.config.get('balances') or [])
        seen_dates.update(b['date'] for b in balances)

        # Add JSON balances:
        for jbcfg in self.config.get('json-balances') or []:
            source = jbcfg['source']
            source = os.path.expanduser(source)
            sources = glob.glob(source)
            for srcname in sources:
                with open(srcname, 'r') as sfh:
                    jdata = simplejson.load(sfh, use_decimal=True)
                    date_key = jbcfg['keys']['date']
                    balance_key = jbcfg['keys']['balance']
                    entries = jdata if isinstance(jdata, list) else [jdata]
                    for entry in entries:
                        date = dateutil.parser.parse(entry[date_key]).date()

                        # YAML balances shall take precedence over JSON balances:
                        if date in seen_dates:
                            logging.debug("{}: Ignoring balance from JSON because of overriding YAML balance for same date: {}".format(self.name, entry))
                            continue

                        seen_dates.add(date)
                        balances.append({
                            'date': date,
                            'balance': entry[balance_key],
                            'remark': 'Loaded from {}'.format(srcname),
                        })
                        logging.debug("{}: Loaded balance from JSON: {}".format(self.name, entry))

        # Sort it
        balances = sorted(balances, key=itemgetter('date'))

        return balances

    # TODO: Clean up mixed use of dicts and SimpleNamespace in balances and
    # checkpoints before adding type hints to this function!
    def _insert_balance_checkpoints(self):
        balances_raw = self._collect_balance_checkpoints()

        """
        Get it right, kids:

        Interpolation point:
            A newly inserted transaction with a certain *value* that aims to
            gradually adjust the effective balance so that it eventually
            matches the one provided in the configuration.
            Being *value* transactions, the interpolation points created for a
            single checkpoint don't depend on each other.
            Nevertheless, the calculation of those values is based on an
            existing balance, so for each set of interpolation points being
            inserted, all balances need to be recomputed before inserting the
            next set.

        Checkpoint:
            A newly inserted transaction with a certain *balance* that
            aims to ensure a specific balance regardless of existing
            transactions.
            After insertion, proper transaction values must be determined based
            on the balance of the existing transactions (including
            interpolation points!), right before the point of checkpoint
            insertion.
        """

        # Transform balance objects from configuration:
        balances = []
        for b in balances_raw:
            # By default, interpolation is disabled. This is because it has the
            # potential to greatly distort the calculation of capital
            # gains and returns (by influencing the assumed invested amount).
            # Use with care!
            interpolate = b['interpolate'] if 'interpolate' in b else False
            if interpolate is True:
                # -1 means heuristic
                interpolate = -1
            balances.append(SimpleNamespace(
                    date=b['date'],
                    balance=Decimal(b['balance']),
                    interpolate=interpolate,
                    remark=b.get('remark') or '',
                    ))

        self._insert_balance_interpolation_points(balances)

        # Prepare list of checkpoint transactions:
        checkpoint_txns = []
        for b in balances:
            t = Transaction(
                date=b.date,
                balance=b.balance,
                subject='Balance checkpoint' + formatifnonempty(': {}', getattr(b, 'remark', '')),
                # Assuming that balance checkpoints already reflect "hidden" changes in wealth:
                isneutral=False,
            )

            # This is a bit hacky but serves the current use case:
            balance_checkpoint_behavior = self.config.get('balance-checkpoint-behavior') or {}
            if 'peeraccount' in balance_checkpoint_behavior:
                t.peeraccount = balance_checkpoint_behavior['peeraccount']
                t.isneutral = None

            checkpoint_txns.append(t)

            logging.debug("{}: Preparing fixed balance checkpoint: {}, balance={}".format(self.name, t.date, t.balance))

        checkpoint_txns.sort(key=attrgetter('date'))

        newlist = []

        # Rebuild transaction list, while weaving in the checkpoint
        # transactions with proper values and recomputing the subsequent
        # balances accordingly. (This is even more complicated than it looks,
        # but don't despair.)
        txnq = deque(checkpoint_txns)
        curbalance = self.initial_balance
        assert curbalance is not None

        # Handle special case: Any checkpoints with earlier dates than the
        # first regular transaction must be inserted first.
        # FIXME: I suspect this whole thing could be implemented in a better way.
        while txnq and (len(self.txns) == 0 or txnq[0].date < self.txns[0].date):
            newt = txnq.popleft()
            newt.value = newt.balance - curbalance
            curbalance = newt.balance
            newlist.append(newt)
            logging.debug("{}: Assigning checkpoint txn value (first txn): {}".format(self.name, newt.shortstr()))

        for idx, txn in enumerate(self.txns):
            curbalance += txn.value
            txn.balance = curbalance
            newlist.append(txn)

            # Insert all checkpoints that fit here
            while txnq and (
                # Already end of list, just keep going through queue:
                (idx + 1 == len(self.txns))
                or
                # Next existing txn is later than top of new-queue, so insert here:
                (self.txns[idx + 1].date > txnq[0].date)
            ):
                # Assign checkpoint txn value based on final balance of (or
                # before) that day:
                newt = txnq.popleft()
                newt.value = newt.balance - curbalance
                curbalance = newt.balance
                newlist.append(newt)

                logging.debug("{}: Assigning checkpoint txn value: {}".format(self.name, newt.shortstr()))

        self.txns = newlist
