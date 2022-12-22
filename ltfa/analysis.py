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

class Analysis():
    def __init__(self, accounts: list, salary_matchers: list) -> None:
        self.classify_savings(accounts)
        self.classify_salary(accounts, salary_matchers)
        self.analyze_capgains(accounts)

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

    def classify_savings(self, accounts) -> None:
        # Consider only non-investment accounts
        accounts = [a for a in accounts if
                    a.meta.config.get('asset-type') != 'investment'
                    and len(a.txns)]

        if not accounts:
            self.savings = None
            return

        # Savings is the daily sum of all non-neutral transactions
        savings = pd.concat([a.txns[a.txns.isneutral == False][['value']] for a in accounts], sort=True)
        savings = savings.groupby(level='date').sum()

        # Assume zero savings for all days without savings (obviously)
        savings = savings.resample('1D').mean().fillna(0)

        self.savings = savings

    def classify_salary(self, accounts, salary_matchers) -> None:
        # Only consider non-investment accounts for incoming salary
        accounts = [a for a in accounts if
                    a.meta.config.get('asset-type') != 'investment'
                    and len(a.txns)]

        if not accounts:
            self.salary = None
            return

        spending = pd.concat([a.txns[a.txns.isneutral == False] for a in accounts], sort=True)

        def is_salary(df, matcher=dict()) -> pd.DataFrame:
            # Return "all false" if matcher has no conditions:
            if matcher == {}:
                return df != df

            # Turn matcher conditions into boolean masks
            masks = []
            for k, v in matcher.items():
                if k == 'peername':
                    masks.append(df.peername.str.contains(v, case=False))
                elif k == 'subject':
                    masks.append(df.subject.str.contains(v, case=False))
                elif k == 'minimum-value':
                    masks.append(df.value >= Decimal(v))
                elif k == 'value':
                    masks.append(df.value == Decimal(v))
                elif k == 'not-before':
                    masks.append(df.index >= pd.to_datetime(v))
                elif k == 'not-after':
                    masks.append(df.index <= pd.to_datetime(v))
                else:
                    raise ValueError(f'Unknown key in salary matcher: {k}')

            # Combine all masks with logical AND
            mask = functools.reduce(lambda a, b: a & b, masks)

            if not mask.any():
                # Consider making this non-fatal if it turns out problematic
                raise ValueError(f'Warning (fatal for now): Salary matcher did not match any transaction: {matcher}')

            return mask

        # Create boolean mask for each matcher
        salary_masks = [is_salary(spending, matcher=m) for m in salary_matchers]

        # Combine masks with logical OR
        if salary_masks:
            salary_mask = functools.reduce(lambda a, b: a | b, salary_masks)
        else:
            salary_mask = spending != spending

        # Apply mask to all spending transactions, get salary dataframe
        salary = spending.where(salary_mask).dropna()

        for row in salary.itertuples():
            logging.debug(f'Classifying as salary: {row}')

        # Keep only required columns (some downstream steps choke on the strings and boolean data):
        salary = salary[['value']]

        self.salary = salary


    def analyze_capgains(self, accounts) -> None:
        # Relevant base metrics for plotting capital gains:
        #  * invested amount (equals balance, already known)
        #  * any gains and losses
        # What to plot:
        #  * for all investment accounts, make a stacked plot of their balance
        #    (Note: any kind of distinguishably drawn plot for neutral/non-neutral txns doesn't really work well, don't try again!)
        #  * overlaid, on the same y-axis, plot the overall yearly returns (absolute value) on a 1y moving window
        #  * overlaid, on a different y-axis, plot the overall return rates on a 1y moving window

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
        #  self.capgains_beginningoftime = min([min(a.txns.index) for a in accounts if a.txns.any()])
        #  self.capgains_endoftime = max([max(a.txns.index) for a in accounts if a.txns.any()])
        self.capgains_beginningoftime = min([a.txns.index.min() for a in accounts])
        self.capgains_endoftime = max([a.txns.index.max() for a in accounts])

        # Compute total invested amount for each day from _all_ txns in _all_
        # accounts.
        # WATCH OUT: This gets massively distorted by balance checkpoint
        # interpolation! Those interpolation points increase the assumed invested
        # amount even though that money was not yet being invested because it was
        # not yet paid out!
        # => Be very careful with using interpolation in the configuration of
        #    investment accounts! Only enable it when you can really assume a
        #    continuous payout which you just don't know the dates of! Don't just
        #    use it to smooth the graphs! When in doubt, assume payout at the end
        #    of an investment.
        # 1. Bring all transactions together and sort them
        totalinvest = pd.concat([a.txns[['value']] for a in accounts], sort=True).sort_index()
        # 2. Keep only one row per day (simply summing up the values)
        totalinvest = totalinvest.groupby(level='date').sum()
        # 3. Build the cumulative sum and call it "balance"
        totalinvest = totalinvest.cumsum().rename(columns={'value': 'totalinvest'})
        # 4. Fill missing data by padding the value (unchanged balance)
        totalinvest = totalinvest.resample('1D').ffill()

        # Combine non-neutral txns (i.e., gains and losses) from all accounts, and
        # make sure the series starts at the "beginning of time", i.e., potentially
        # before the first actual gain.
        gains_beginning = pd.DataFrame([(totalinvest.index[0], float(0))], columns=['date', 'value']).set_index('date')
        gains = pd.concat([gains_beginning] + [a.txns[a.txns.isneutral == False][['value']] for a in accounts], sort=True)
        gains = gains.rename(columns={'value': 'gains'})
        gains = gains.groupby(level='date').sum()

        # Before upsampling, keep record of the first actual gain (before
        # upsampling). This is useful for plotting, since the data before that
        # is likely distorted by the investement balance average "settling
        # down".
        self.date_of_first_actual_gain = gains[gains.gains > 0].index[0]

        # Upsample to daily values (missing data will be NaN)
        gains = gains.resample('1D').mean()

        # Redistribute gains evenly over all days. The assumption is, that a
        # particular gain can be attributed to all NaN-days coming up to it. Note
        # that this approach is very different to interpolating balance checkpoints
        # early in the pipeline, which can distort our calculations of the invested
        # amount. Just distributing the gains like this appears to be the only way
        # to get useful EWMH plots out of the data.
        gains = gains.cumsum().interpolate(method='time').diff()

        # Daily balance of invested money (total on that day)
        self.totalinvest = totalinvest

        # Daily amount of capital gains (earned on that day only)
        self.gains = gains


    def print_capgains_analysis(self) -> None:
        for i in (8, 4, 2):
            ret = self.get_averaged_capgains(ewm_span_years=i)

            logging.info("{}y ewm final values:".format(i))
            logging.info("\tinvested     = {:.0f} €".format(ret.totalinvest.totalinvest[-1]))
            logging.info("\tgains p.a.   = {:.0f} €".format(ret.gainspa.gains[-1]))
            logging.info("\treturns p.a. = {:.2%}".format(ret.returns.returns[-1]))

        # Overall results (annualized) up to $now:
        overall = self.get_averaged_capgains()

        period = daycount_tostring((self.capgains_endoftime - self.capgains_beginningoftime).days)
        logging.info("Statistics over all time ({}):".format(period))
        logging.info("\tAvg. invested amount: {:.0f} €".format(overall.totalinvest.totalinvest[-1]))
        logging.info("\tTotal capital gains: {:.0f} €".format(self.gains.gains.sum()))
        logging.info("\tAvg. capital gains (p.a.): {:.0f} €".format(overall.gainspa.gains[-1]))
        # Note: Compounding is implicitly part of this metric because we look
        # at the actual avg. invested amount.
        logging.info("\tAvg. returns (p.a.): {:.2%}".format(overall.returns.returns[-1]))


        # Discrete, cumulated stats for each calendar year ('A' means year-end):
        yearlyreturns = self.gains.resample('A').sum().merge(self.totalinvest.resample('A').mean(),
                                                              left_index=True, right_index=True)
        for date, item in yearlyreturns.iterrows():
            if date > pd.Timestamp(self.capgains_endoftime):
                # Skip current year if it hasn't concluded yet (see stats for past
                # year instead)
                continue

            if date - datetime.timedelta(days=365) < pd.Timestamp(self.capgains_beginningoftime):
                # Skip incomplete first year
                continue

            gains = item['gains']
            invested = item['totalinvest']
            logging.info("Return rate in {}: {:.2%} ({:.0f} € gains on {:.0f} € invested)".format(
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

            logging.info('Return rate past 1y: {:.2%} ({:.0f} € gains on {:.0f} € invested)'.format(
                gains_yoy / invested_yoy, gains_yoy, invested_yoy
                ))
        stats_for_yoy()
