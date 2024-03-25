from ltfa.util import daycount_tostring
from types import SimpleNamespace
import itertools
from decimal import Decimal
import logging
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import re
import functools
from typing import Optional
from operator import attrgetter
from collections import defaultdict

class Analysis():
    def __init__(self, accounts: list, salary_matchers: list) -> None:
        for account in accounts:
            self.classify_salary(account.txns, salary_matchers)

        self.txns = pd.concat([a.txns for a in accounts], sort=True).sort_index()

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            self.debug_log_salary()

        self.daily_savings = self.compute_daily_savings(accounts)

        self.salary = self.txns[self.txns['salary']][['value']]

        self.analyze_capgains(accounts)

    def debug_log_salary(self):
        for txn in self.txns[self.txns.salary][['value', 'subject', 'peername', 'account']].itertuples(name='Salary'):
            txn = txn._replace(Index = str(txn.Index.date()))
            logging.debug(txn)

    def get_averaged_capgains(self, ewm_span_years: Optional[float] = None) -> SimpleNamespace:
        if ewm_span_years:
            mangler = lambda x: x.ewm(span=ewm_span_years * 365).mean()
        else:
            # No EWM requested, just build the overall average
            mangler = lambda x: x.expanding().mean()

        # Apply mangler (there shouldn't be any NaN values, but fill them with
        # zero just to be sure):
        gains = mangler(self.gains.fillna(0))
        totalinvest = mangler(self.totalinvest.fillna(0))

        # Annualize gains
        gainspa = gains * 365

        # Compute actual return rates (annualized)
        returns = gainspa.merge(totalinvest, left_index=True, right_index=True)
        returns['returns'] = returns.gains / returns.totalinvest

        return SimpleNamespace(
                totalinvest=totalinvest,
                gainspa=gainspa,
                returns=returns,
                )

    def compute_daily_savings(self, accounts):
        # We define savings as the sum of all non-neutral transactions that
        # haven't happened on transaction accounts (because those would be
        # capital gains). Treating savings as a type of transaction (in order,
        # e.g., to list them in a report) like we do it for salary or spending
        # would not really be useful as it is rather just the amount of money
        # *not spent* that is interesting. Therefore, we're building the daily
        # sums here and abstract away from the individual transactions.

        return self.txns[~self.txns.isneutral & (self.txns.asset_type != 'investment')][['value']].resample('1D').sum().fillna(0)

    def classify_salary(self, df, salary_matchers):
        # Start out with all-false mask
        combined_mask = df.index != df.index

        for matcher in salary_matchers:
            # Skip empty matchers
            if matcher == {}:
                next

            # Accumulate matcher conditions with logical AND, starting with an
            # all-true mask:
            mask = df.index == df.index
            for k, v in matcher.items():
                if k == 'peername':
                    mask &= df.peername.str.contains(v, case=False)
                elif k == 'peeraccount':
                    mask &= df.peeraccount == v
                elif k == 'subject':
                    mask &= df.subject.str.contains(v, case=False)
                elif k == 'minimum-value':
                    mask &= df.value >= Decimal(v)
                elif k == 'value':
                    mask &= df.value == Decimal(v)
                elif k == 'not-before':
                    mask &= df.index >= pd.to_datetime(v)
                elif k == 'not-after':
                    mask &= df.index <= pd.to_datetime(v)
                else:
                    raise ValueError(f'Unknown key in salary matcher: {k}')

            combined_mask |= mask

        df['salary'] = False
        df.loc[combined_mask, 'salary'] = True

    def analyze_capgains(self, accounts) -> None:
        # Only consider non-empty, investment-type accounts from here onwards:
        accounts = [a for a in accounts if
                    a.meta.config.get('asset-type') == 'investment'
                    and len(a.txns)]

        self.has_capgains = len(accounts) > 0
        if not self.has_capgains:
            logging.info('No investment accounts defined, cannot make capgains plot')
            # TODO: Change to this message (needs update in tests!)
            #  logging.info('No investment accounts defined, cannot analyze capgains')
            return

        # Determine date range of investment data
        self.capgains_beginningoftime = min([a.txns.index.min() for a in accounts])
        self.capgains_endoftime = max([a.txns.index.max() for a in accounts])

        # Compute invested amounts and gains on a per-account basis. Doing this
        # separately is essential for the gains interpolation to work (see below).

        all_invest = []
        all_gains = []
        for account in accounts:
            # Compute total invested amount for each day.
            # WATCH OUT: This gets massively distorted by balance checkpoint
            # interpolation! Those interpolation points increase the assumed invested
            # amount even though that money was not yet being invested because it was
            # not yet paid out!
            # => Be very careful with using interpolation in the configuration of
            #    investment accounts! Only enable it when you can really assume a
            #    continuous payout which you just don't know the dates of! Don't just
            #    use it to smooth the graphs! When in doubt, assume payout at the end
            #    of an investment.

            # Invest: Start with all txn values and make sure the series is
            # extended as far as the most recent investment balance globally.
            # This makes sure that it contributes to the total sum even though
            # it hasn't been changed in a while.
            invest = account.txns[['value']]
            invest_endoftime = pd.DataFrame([(self.capgains_endoftime, float(0))], columns=['date', 'value']).set_index('date')
            invest = pd.concat([invest, invest_endoftime], sort=True).sort_index()
            # Sum up values per day (eliminates possible duplicate zero-value entries from loading)
            invest = invest.groupby(level='date').sum()
            # Build cumulative sum, i.e., balance
            invest = invest.cumsum().rename(columns={'value': 'invest'})

            # Fill missing data by padding the value (same balance as previous day)
            invest = invest.resample('1D').ffill()

            # Gains: Gather all non-neutral txns and make sure the series starts
            # exactly when the balance first is greater than zero, since that is
            # when the investment begins (even if the first gain comes much later)
            gains_beginning = pd.DataFrame([(invest[invest.invest > 0].index[0], float(0))], columns=['date', 'value']).set_index('date')
            gains = pd.concat([gains_beginning] + [account.txns[account.txns.isneutral == False][['value']]], sort=True)
            gains = gains.rename(columns={'value': 'gains'})
            gains = gains.groupby(level='date').sum()

            # Upsample to daily values (missing data will be NaN)
            gains = gains.resample('1D').mean()

            # Redistribute gains evenly over all days. The assumption is that a
            # particular gain can be attributed to all NaN-days leading up to it. Note
            # that this approach is very different to interpolating balance checkpoints
            # early in the pipeline, which can distort our calculations of the invested
            # amount. Just distributing the gains like this appears to be the only way
            # to get useful EWMH plots out of the data.
            gains_redistributed = gains.cumsum().interpolate().diff()

            # Calling diff() inherently discards the very first value (sets it
            # to zero), but it's still a gain that needs to be included, so we
            # restore it manually here:
            gains_redistributed.iloc[0] = gains.iloc[0]
            gains = gains_redistributed

            all_invest.append(invest)
            all_gains.append(gains)

        totalinvest = pd.concat(all_invest, sort=True).sort_index()
        totalinvest = totalinvest.groupby(level='date').sum().rename(columns={'invest': 'totalinvest'})

        gains = pd.concat(all_gains, sort=True).sort_index()
        gains = gains.groupby(level='date').sum()

        # Daily balance of invested money (total for each particular day)
        self.totalinvest = totalinvest

        # Daily amount of capital gains (what was gained on each particular day)
        self.gains = gains


    def make_capgains_analysis(self, fh) -> None:
        def p(s):
            return print(s, file=fh)

        for i in (8, 4, 2):
            ret = self.get_averaged_capgains(ewm_span_years=i)

            p("{}y ewm:".format(i))
            p("\tinvested     = {:.0f} €".format(ret.totalinvest.totalinvest.iloc[-1]))
            p("\tgains p.a.   = {:.0f} €".format(ret.gainspa.gains.iloc[-1]))
            p("\treturns p.a. = {:.2%}".format(ret.returns.returns.iloc[-1]))

        # Overall results (annualized) up to $now:
        overall = self.get_averaged_capgains()

        period = daycount_tostring((self.capgains_endoftime - self.capgains_beginningoftime).days)
        p("Statistics over all time ({}):".format(period))
        p("\tAvg. invested amount: {:.0f} €".format(overall.totalinvest.totalinvest.iloc[-1]))
        p("\tTotal capital gains: {:.0f} €".format(self.gains.gains.sum()))
        p("\tAvg. capital gains (p.a.): {:.0f} €".format(overall.gainspa.gains.iloc[-1]))
        # Note: Compounding is implicitly part of this metric because we look
        # at the actual avg. invested amount.
        p("\tAvg. returns (p.a.): {:.2%}".format(overall.returns.returns.iloc[-1]))


        # Discrete, cumulated stats for each calendar year ('A' means year-end):
        yearlyreturns = self.gains.resample('A').sum().merge(self.totalinvest.resample('A').mean(),
                                                              left_index=True, right_index=True)
        longest_return_rate_string = max(len("{:.2%}".format(item['gains'] / item['totalinvest'])) for _, item in yearlyreturns.iterrows())
        for date, item in yearlyreturns.iterrows():
            if date > pd.Timestamp(self.capgains_endoftime):
                # Skip current year if it hasn't concluded yet (see stats for past
                # year instead)
                continue

            first_year_is_incomplete = not (self.capgains_beginningoftime.day == 1 and self.capgains_beginningoftime.month == 1)
            if first_year_is_incomplete and date.year == self.capgains_beginningoftime.year:
                # Do not show yearly returns for the incomplete first year
                continue

            gains = item['gains']
            invested = item['totalinvest']
            p(('Return rate in {}: {:' + str(longest_return_rate_string) + '.2%} ({:.0f} € gains on {:.0f} € invested)').format(
                date.year, gains / invested, gains, invested
                ))

        # Stats for the past year (as in: ~365 days)
        def stats_for_yoy() -> None:
            # Calculate start date for "one year before end-of-time". We use
            # dateutil.relativedelta to jump to the same date a year before (not
            # just 365 days ago) and then add a day because range slicing is
            # inclusive:
            one_year_before = self.capgains_endoftime - relativedelta(years=1) + relativedelta(days=1)

            invested_yoy = self.totalinvest[one_year_before : self.capgains_endoftime].mean().totalinvest
            gains_yoy = self.gains[one_year_before : self.capgains_endoftime].sum().gains

            p(('Return rate past 1y: {:' + str(longest_return_rate_string) + '.2%} ({:.0f} € gains on {:.0f} € invested)').format(
                gains_yoy / invested_yoy, gains_yoy, invested_yoy
                ))
        stats_for_yoy()

    def make_monthly_overview(self, fh, month):
        spending = self.txns[~self.txns.salary & ~self.txns.isneutral & (self.txns.asset_type != 'investment')].loc[month]

        capgains = Decimal(self.txns[~self.txns.isneutral & (self.txns.asset_type == 'investment')].loc[month].value.sum()).quantize(Decimal('1'))
        total_spending = -Decimal(spending.value.sum()).quantize(Decimal('1'))

        value_format = '.2f'
        longest_value = max(len(f'{v:{value_format}}') for v in spending.value)

        def nicify(s):
            # Titleize any all-caps words at least 5 chars long
            s = re.sub(r'\b[A-Z/+-]{5,999}\b', lambda m: m.group(0).title(), s)
            return s

        def find_truncate_length(allstrings, maxlen):
            longest = max(len(s) for s in allstrings)
            return min(longest, maxlen)

        def trunc_align(s, maxlen):
            s = s[:maxlen - 3] + '...' if len(s) > maxlen else s
            s = re.sub(r'\.\.\.+$', '...', s)
            return f'{s: <{maxlen}}'

        max_peername_length = find_truncate_length(spending.peername, 25)
        max_subject_length = find_truncate_length(spending.subject, 120)

        spending_strs_by_account = defaultdict(list)
        for t in sorted(spending.itertuples(), key=attrgetter('value')):
            subject = nicify(t.subject)
            subject = trunc_align(subject, max_subject_length)
            peername = nicify(t.peername)
            peername = trunc_align(peername, max_peername_length)
            spending_strs_by_account[t.account].append(f'{t.value: >{longest_value}{value_format}} €  {peername}  {subject}')

        def p(s=''):
            """Print-to-file helper"""
            return print(s, file=fh)

        p(f'Overview for {month}')
        p(f'====================')
        p()
        p(f'Total spending: {total_spending} €')
        p(f'Capital gains: {capgains} €')
        p()
        for account, lines in spending_strs_by_account.items():
            p()
            header = f'Spending transactions: {account}'
            p(header)
            p('-' * len(header))
            p('\n'.join(lines))
