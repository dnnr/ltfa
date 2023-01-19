import bokeh.plotting as bkp
import bokeh as bk
import pandas as pd
import functools
import numpy as np
import itertools
import matplotlib.colors
import logging
import datetime
from types import SimpleNamespace
from ltfa.util import color_scale_lightness
from typing import Generator

def color_gen() -> Generator:
    yield from itertools.cycle(bk.palettes.Category10[10])


def custom_hovertool_formatter() -> bk.models.tools.CustomJSHover:
    """ Universal formatting helper that covers the various representation
    needs of the tooltips. """
    return bk.models.tools.CustomJSHover(code="""
            if (format == "display_none_if_nan") {
                return isNaN(value) ? "display:none;" : ""
            } else if (format == "sign_color") {
                if (value < 0) {
                    return "color: red;"
                } else if (value > 0) {
                    return "color: green;"
                }
                return ""
            } else if (format == "currency") {
                return value.toLocaleString("en-US", {"minimumFractionDigits": 2, "maximumFractionDigits": 3})
            } else if (format != "" && !isNaN(value)) {
                return sprintf(format, value)
            } else {
                return value
            }
        """)


def stack_dataframes(accounts_df) -> list[pd.DataFrame]:
    """ Take list of accounts and dataframes and stack their txn values. """

    # First order accounts by their activity (ascending), which yields a
    # visually optimal stacking
    accounts_df = list(sorted(accounts_df, key=lambda a: len(a.dailies)))

    # Initialize the accumulator with the first account as-is
    ret = accounts_df[:1]

    # Initialize first account's "bottom" values with all zeroes
    ret[0].dailies['bottom'] = 0.

    # Initialize first account's "top" values with just its balance
    ret[0].dailies['top'] = ret[0].dailies.value.cumsum()

    for account_df in accounts_df[1:]:
        prev_account = ret[-1]

        # Insert all previous "top" values into the current transaction list as its "bottom" column (using an "outer join", because the dates of each side might be different)
        prev_top_as_bottom = prev_account.dailies.drop('bottom', axis=1).rename(columns={'top': 'bottom'}).bottom
        account_df.dailies = account_df.dailies.join(prev_top_as_bottom, how='outer')

        # Fill NaNs in "bottom" (i.e. all dates that only appear in the current
        # list) by padding the previous value [might not even be necessary?]
        account_df.dailies.bottom.fillna(method='pad', inplace=True)

        # Compute new "top" by adding the "bottom" and the cumsum of new transactions
        #  account_df.dailies['top'] = (prev_account.dailies.value + account_df.dailies.value).cumsum()
        #  account_df.dailies['top'] = prev_account.dailies.value.add(account_df.dailies.value, fill_value=0).cumsum()
        account_df.dailies['top'] = account_df.dailies.bottom.add(account_df.dailies.value.fillna(0).cumsum(), fill_value=0)


        #  print(f'stacked (before resampling):\n{account_df.dailies}')
        #  account_df.dailies = account_df.dailies.reindex(account_df.dailies.resample('1D').sum().index)

        #  account_df.dailies = account_df.dailies.mask(account_df.dailies.shift(1) == account_df.dailies).dropna(how='all')


        # Try dropping repeated values (to reduce number of drawn glyphs)
        #  for window in account_df.dailies.rolling(window=3):
            #  if len(window) == 3 and window.top.nunique() == 1:
                #  print(f"window:\n{window}")
                #  account_df.dailies.loc[window.index[1]].top = np.nan
                #  print(f"window after nanning:\n{window}")
                #  sys.exit(1)
                #  pass

        #  print(f'stacked:\n{account_df.dailies}')

        ret.append(account_df)

    def fake_step_post_maker(df):
        """
        Fake a step-wise plot as a workaround for bokeh not supporting
        step-wise vareas (see https://github.com/bokeh/bokeh/issues/12062): For
        each entry, insert another one on the day right before (unless it
        already exists) to repeat/pad the previous value.
        """
        for window in df.rolling(window=2, min_periods=0):
            yield (window.index[0], window.iloc[0])
            # Only consider full windows
            if len(window) != 2:
                continue

            day_before = window.index[1] - datetime.timedelta(days=1)
            # If the date already has a row, we cannot (need not) insert a fake row
            if day_before != window.index[0]:
                # Copy the first item in the window, but clear the "value" column:
                fake_in = window.iloc[0].copy()
                fake_in.value = np.nan
                yield (day_before, fake_in)
        else:
            # The very last row:
            yield (window.index[1], window.iloc[1])

    for account in ret:
        df_step_post = pd.DataFrame.from_dict(dict(fake_step_post_maker(account.dailies)), orient='index')
        df_step_post.index.name = 'date'
        account.dailies = df_step_post

    return ret


def add_balances_plot(figure, accounts, accounts_stacked, annotations, analysis) -> None:
    marker_glyphs = []
    colors = color_gen()
    for account in accounts_stacked:
        # Also compute a lighter color for the tooltip circles so that the line
        # itself (drawn on top) remains visible even if there are many
        # overlapping circles. Note that this can be done easier (see commented
        # out code) but before bokeh-3.0 that triggers an HSL() deprecation
        # warning, see https://github.com/bokeh/bokeh/issues/11845.
        # Easier variant, doesn't need util function:
        #  lighter_color = bk.colors.RGB(*[x * 255 for x in matplotlib.colors.to_rgb(this_color)]).lighten(0.2)
        #  lighterer_color = bk.colors.RGB(*[x * 255 for x in matplotlib.colors.to_rgb(this_color)]).lighten(0.3)

        this_color = next(colors)
        lighter_color = bk.colors.RGB(*[int(x * 255) for x in color_scale_lightness(matplotlib.colors.to_rgb(this_color), 1.4)])
        lighterer_color = bk.colors.RGB(*[int(x * 255) for x in color_scale_lightness(matplotlib.colors.to_rgb(this_color), 1.55)])

        dailies = account.dailies

        # Note: Stuff I figured out about what data ends up in the HTML output:
        #   * Columns that are not used for any glyphs/tooltips will still be exported

        # For each transaction, add the corresponding stack top value for that
        # day (-> y-position of the marker). Also, ignore any zero-value
        # transaction here, because they're simply not worth a marker/tooltip.
        txns_with_top = account.txns[account.txns.value != 0].join(dailies[['top']], how='left')

        # Select daily balance items to be represented in tooltips. We want
        # every visible transaction to be accompanied with a balance value. Any
        # other items in "dailies" would be just noise or helper rows for
        # step-wise drawing.
        balances_for_markers = dailies[dailies.index.isin(txns_with_top.index)][['balance', 'top']]

        # Add the account name to balance row so that it can be included in the tooltip
        balances_for_markers['account'] = account.meta.name

        # Combine daily balance values and transactions in a single frame so
        # that we can draw their markers in a single command. This is the only
        # way to make bokeh merge their tooltips instead arbitrarily
        # overlapping them.
        balances_and_txns = pd.concat([balances_for_markers, txns_with_top])

        # Bokeh chokes on boolean columns (fails in JSON serialization). We
        # don't need this column anyway, so drop it:
        balances_and_txns.drop('isneutral', axis=1, inplace=True)

        # concat() doesn't keep the index sorted, so do that explicitly. Use a
        # stable sorting here to ensure balance row always appears before the
        # txn rows in the tooltip:
        balances_and_txns.sort_index(inplace=True, kind='stable')

        balances_and_txns['account'] = account.meta.name

        # Prepare a marker alpha column that is non-zero for every first row
        # per day and zero for the rest, so that only one marker (circle) is
        # visible per day, but all of them contribute to the tooltip content:
        balances_and_txns['marker_alpha'] = [0 if isdup else 0.7 for isdup in balances_and_txns.index.duplicated()]

        # Finally draw the markers
        # TODO: We should draw them in a single call for all accounts to avoid
        # remaining overlapping artifacts in tooltips (likely to happen when an
        # account goes to zero)
        marker_glyphs += [figure.circle(source=balances_and_txns, x='date', y='top', color=lighterer_color, line_color=lighter_color, fill_alpha='marker_alpha', line_alpha='marker_alpha', size=8, legend_label=account.meta.name)]

        # Draw lines and areas only for regions where the account has non-zero
        # value (but use expand_mask so that the first and last zero-value
        # entries are included, otherwise we'll get weird drawing artifacts)
        valid_values_mask = expand_mask(dailies.bottom != dailies.top)
        figure.varea(source=dailies.where(valid_values_mask), x='date', y1='bottom', y2='top', color=this_color, fill_alpha=0.2, legend_label=account.meta.name)
        figure.line(source=dailies.where(valid_values_mask), x='date', y='top', color=this_color, line_width=1, legend_label=account.meta.name)

    # The tooltips are a bit hacky: We draw the same one for every marker
    # (balances and every transaction), but use custom formatter to hide the
    # parts that are not relevant.
    # Note that unlike in the stock tooltips, we cannot use real CSS tables here
    # because bokeh insists on wrapping each tooltip in two DIVs when
    # concatenating overlapping onces, which prohibits the use of an actual
    # table. An alternative would be to put all transactions into one row, but
    # that would mean parsing it (as JSON?) in the formatter. I'd rather not.
    # Also note that we rely on custom global CSS here (see the save() call at
    # the end).
    TOOLTIPS = """
        <div class="bk-tooltip-date" style="@balance{display_none_if_nan}">
            <span class="bk bk-tooltip-row-label" style=" font-weight: bold;">Date: </span>
            <span class="bk bk-tooltip-row-value" style="">
                <span class="bk">@date{%F (%a)}</span>
            </span>
        </div>
        <div style="@balance{display_none_if_nan}">
            <span class="bk bk-tooltip-row-label" style=" font-weight: bold;">Balance: </span>
            <span class="bk bk-tooltip-row-value" style="">
                <span class="bk">€ @balance{currency} (@account)</span>
            </span>
        </div>
        <div class="bk-tooltip-txn" style="@value{display_none_if_nan}">
            <span class="bk bk-tooltip-row-value" style="display: table-cell;">
                <span style="@value{sign_color}">€ @value{currency}</span> @peername (@subject)
            </span>
        </div>
    """

    # All tooltips have to originate from a single HoverTool instance,
    # otherwise multiple tooltips will be rendered and overlap. The distinction
    # between balance and txn tooltips has to happen within the tooltip.
    figure.add_tools(bk.models.HoverTool(renderers=marker_glyphs,
                                         toggleable=False,
                                         formatters={
                                             '@date': 'datetime',
                                             '@balance': custom_hovertool_formatter(),
                                             '@value': custom_hovertool_formatter(),
                                         },
                                         tooltips=TOOLTIPS,
                                         ))

    annotations_guideline = accounts_stacked[-1].dailies[['top']].rename(columns={'top': 'value'})
    if 'balance' in annotations:
        add_annotations(figure, annotations['balance'], annotations_guideline)
    if 'spending' in annotations:
        add_annotations(figure, annotations['spending'], annotations_guideline)


def add_annotations(figure, annotations, guideline, y_offset_factor=0.05, color='black', fill_alpha=0.25, legend_label=None) -> None:
    """ Add annotations. The guideline has to have a column named 'value'. """

    # Interpolate to ensure that we have values for every day to attach the
    # annotations to:
    guideline = guideline.resample('1D').interpolate()

    # Compute a windowed max value of the guideline to ensure that the triangles
    # don't collide with the data. Since the triangles are not fixed to the
    # data coordinate system (i.e., they "grow larger" when zooming out), this
    # is not a 100% reliable, but should be good enough.
    guideline = guideline.rolling(14, center=True, min_periods=1).max()

    # Add a relative vertical offset to the triangles (based on the y-range of the data):
    guideline += guideline.max() * y_offset_factor

    annotations = pd.DataFrame.from_dict(annotations, orient='index', columns=['text'])
    annotations = annotations.join(guideline, how='left')

    maybe_legend_label = {'legend_label': legend_label} if legend_label else {}
    annotations_glyph = figure.inverted_triangle(source=annotations, x='index', y='value', color=color, fill_alpha=fill_alpha, size=20, **maybe_legend_label)

    figure.add_tools(bk.models.HoverTool(renderers=[annotations_glyph],
                                         toggleable=False,
                                         formatters={
                                             '@index': 'datetime',
                                         },
                                         tooltips=[
                                             ('Date', '@index{%F (%a)}'),
                                             ('Text', '@text'),
                                         ]
                                         ))

def add_invest_and_gains_plot(figure, accounts, annotations, analysis) -> None:
    invested_label = 'Invested amount (currently)'
    avg_invested_label = 'Invested amount (all-time average)'
    gains_label = 'Capital gains (cumulative)'

    invested_glyph = figure.line(source=analysis.totalinvest, x='date', y='totalinvest', legend_label=invested_label, line_width=1.3, color='black')
    avg_invested_glyph = figure.line(source=analysis.get_averaged_capgains().totalinvest, x='date', y='totalinvest', legend_label=avg_invested_label, line_width=1.3, color='black', line_dash='dashed')
    gains_glyph = figure.line(source=analysis.gains.cumsum(), x='date', y='gains', legend_label=gains_label, line_width=1.3, color='green')

    figure.add_tools(bk.models.HoverTool(renderers=[invested_glyph],
                                         toggleable=False,
                                         formatters={
                                             '@date': 'datetime',
                                             '@totalinvest': custom_hovertool_formatter(),
                                         },
                                         tooltips=[
                                             ('Date', '@date{%F (%a)}'),
                                             (invested_label, '€ @totalinvest{currency}'),
                                         ]
                                         ))

    figure.add_tools(bk.models.HoverTool(renderers=[avg_invested_glyph],
                                         toggleable=False,
                                         formatters={
                                             '@date': 'datetime',
                                             '@totalinvest': custom_hovertool_formatter(),
                                         },
                                         tooltips=[
                                             ('Date', '@date{%F (%a)}'),
                                             (avg_invested_label, '€ @totalinvest{currency}'),
                                         ]
                                         ))

    figure.add_tools(bk.models.HoverTool(renderers=[gains_glyph],
                                         toggleable=False,
                                         formatters={
                                             '@date': 'datetime',
                                             '@gains': custom_hovertool_formatter(),
                                         },
                                         tooltips=[
                                             ('Date', '@date{%F (%a)}'),
                                             (gains_label, '€ @gains{currency}'),
                                         ]
                                         ))

def add_capital_returns_plot(figure, accounts, annotations, analysis) -> None:
    def plotreturns(figure, analysis, ewm_span_years, alpha, color, line_width, legend_label, with_varea, with_hover) -> pd.DataFrame:
        """ Helper function that plots returns while using a given EWM span (None means overall average) """
        returns = analysis.get_averaged_capgains(ewm_span_years=ewm_span_years).returns
        returns = returns[returns.index >= analysis.date_of_first_actual_gain]

        if with_varea:
            figure.varea(source=returns, x='date', y1=0, y2='returns', color=color, fill_alpha=0.5 * alpha, legend_label=legend_label)
        returns_glyph = figure.line(source=returns, x='date', y='returns', color=color, legend_label=legend_label, line_width=line_width, alpha=alpha)

        if with_hover:
            figure.add_tools(bk.models.HoverTool(renderers=[returns_glyph],
                                                 toggleable=False,
                                                 mode='vline',
                                                 formatters={
                                                     '@date': 'datetime',
                                                 },
                                                 tooltips=[
                                                     ('Date', '@date{%F (%a)}'),
                                                     ('Returns (% p.a.)', '@returns{0.00%}'),
                                                 ]
                                                 ))

        return returns

    returnsplotter = functools.partial(plotreturns, figure, analysis, line_width=1.3, color='green')

    # Keep track of all plotted data for annotations
    all_plotted_data = []

    # Area plot with all-time returns
    all_plotted_data.append(returnsplotter(
        ewm_span_years=None,
        alpha=1,
        with_varea=True,
        with_hover=True,
        legend_label=f'all-time',
    ))

    # Returns at several window sizes (smaller windows at more transparency)
    for idx, years in enumerate([8, 4, 2]):
        all_plotted_data.append(returnsplotter(
            ewm_span_years=years,
            alpha = 1 - 0.3 * idx,
            with_varea=False,
            with_hover=False,
            legend_label=f'{years} year EWM window',
        ))

    # Find the "top edge" of all plotted data to use as guideline for the annotations:
    annotations_guideline = pd.concat(all_plotted_data).groupby(level=0).max().rename(columns={'returns': 'value'})
    if 'capgains' in annotations:
        add_annotations(figure, annotations['capgains'], annotations_guideline, 0.15)


def add_spending_and_savings_plot(figure, annotations, analysis) -> None:
    ewm_years_shortterm = 0.5
    ewm_years_midterm = 2
    ewm_years_longterm = 7

    colors = color_gen()
    salary_color = next(colors)
    spending_color = next(colors)
    savings_color = next(colors)
    spending_longterm_color = next(colors)

    all_plotted_data = []

    if analysis.daily_savings is None:
        # Really nothing to plot
        return

    savings_ewm, salary_ewm, salary_monthly_sum = calc_spending_and_salary_ewms(analysis, ewm_years_midterm)

    df = analysis.all_in_one_df
    spending_daily = - df[~df.isneutral & (df.asset_type != 'investment') & ~df.salary][['value']].resample('1D').sum().fillna(0)
    spending_ewm = ewm_daily_as_monthly(spending_daily, ewm_years_shortterm)
    spending_ewm_longterm = ewm_daily_as_monthly(spending_daily, ewm_years_longterm)

    if not analysis.salary.empty:
        figure.line(source=salary_monthly_sum, x='date', y='value', legend_label='Actual salary per month', color=salary_color, line_width=1.3)
        figure.circle(source=salary_monthly_sum, x='date', y='value', size=6, color=salary_color, legend_label='Actual salary per month', fill_alpha=0)
        all_plotted_data.append(salary_monthly_sum)

        figure.line(source=salary_ewm, x='date', y='value', legend_label=f'Salary ({ewm_years_midterm}y EWM)', color=salary_color, line_width=2)
        all_plotted_data.append(salary_ewm)

        figure.line(source=spending_ewm, x='date', y='value', color=spending_color, legend_label=f'Monthly spending ({ewm_years_shortterm}y EWM)', line_width=1.3)
        figure.varea(source=spending_ewm, x='date', y1=0, y2='value', color=spending_color, legend_label=f'Monthly spending ({ewm_years_shortterm}y EWM)', fill_alpha=0.5)
        all_plotted_data.append(spending_ewm)

        figure.line(source=spending_ewm_longterm, x='date', y='value', color=spending_longterm_color, legend_label=f'Estimated monthly spending ({ewm_years_longterm}y EWM)', line_width=1.1)
        all_plotted_data.append(spending_ewm_longterm)

    figure.line(source=savings_ewm, x='date', y='value', color=savings_color, legend_label='Monthly savings (2y EWM)', line_width=1.3, line_alpha=0.8)
    all_plotted_data.append(savings_ewm)

    # Find the "top edge" of all plotted data to use as guideline for the annotations:
    annotations_guideline = pd.concat([df.resample('1D').interpolate() for df in all_plotted_data]).groupby(level=0).max()
    if 'spending' in annotations:
        add_annotations(figure, annotations['spending'], annotations_guideline, 0.15, color=spending_color, legend_label='Spending annotations')
    if 'salary' in annotations:
        add_annotations(figure, annotations['salary'], annotations_guideline, 0.15, color=salary_color, legend_label='Salary annotations')


def ewm_daily_as_monthly(df, years):
    return df.ewm(span=years * 365, min_periods=30).mean().dropna() * 30


def calc_spending_and_salary_ewms(analysis, ewm_years):
    """ Helper to compute spending and salary data using a given EWM span """
    # Compute EWM over savings (suppressing the first 30 days, which are usually somewhat distorted/overweighted)
    savings_ewm = analysis.daily_savings.ewm(span=ewm_years * 365, min_periods=30).mean().dropna() * 30

    # Everything other than savings depends on having salary information:
    salary = analysis.salary
    if not salary.empty:
        # Cut current month from salary date because it's most likely incomplete (distorting month sums)
        salary = salary[salary.index < salary.index[-1].to_period('M').to_timestamp()]

        # Compute plain monthly sum to actually plot as-is
        salary_monthly_sum = salary.resample('1M').sum()

        # Compute EWM of salary, but using a 1M resampling because salary really
        # has a known monthly cadence, and otherwise, we'd just see a saw tooth
        # pattern but no additional information:
        salary_ewm = salary.resample('1M').sum().fillna(0).ewm(span=ewm_years * 365 / 30).mean()

        return savings_ewm, salary_ewm, salary_monthly_sum

    return savings_ewm, None, None


def expand_mask(mask) -> pd.DataFrame:
    """ Helper that takes a bool selection mask and adds all non-selected items
    that are next to selected ones into the selection. """
    ret = mask.copy()
    for idx, value in enumerate(mask):
        if idx > 0 and ret[-1]:
            ret.iloc[idx] = True
        elif idx + 1 < len(mask) and mask[idx + 1]:
            ret.iloc[idx] = True
    return ret


def prepare_figure(title, y_axis_label, y_tick_format) -> bk.plotting.figure:
    common_figure_kwargs = dict(
        tools="pan,box_zoom,reset,save,wheel_zoom",
        active_scroll="wheel_zoom",
        x_axis_type="datetime",
        sizing_mode="stretch_width",
    )

    figure = bkp.figure(title=title, x_axis_label="Date", y_axis_label=y_axis_label, **common_figure_kwargs)

    figure.add_layout(bk.models.LinearAxis(), 'right')
    figure.yaxis.formatter = bk.models.formatters.NumeralTickFormatter(format=y_tick_format)

    figure.xaxis.formatter=bk.models.formatters.DatetimeTickFormatter(
        days="%Y-%m-%d",
        months="%b %Y"
    )

    return figure


def makeplot_balances(accounts, annotations, analysis, file) -> None:
    # Note: NumeralTickFormatter doesn't support currency signs other than $, unfortunately. (bokeh-2.4.2)
    figure1 = prepare_figure(title='Balance', y_axis_label='Amount', y_tick_format='0a')
    figure2 = prepare_figure(title='Capital returns (% p.a.)', y_axis_label='Percentage', y_tick_format='0.00%')
    figure3 = prepare_figure(title='Spending & Savings', y_axis_label='Amount', y_tick_format='0')

    # Synchronize viewports in all figures
    figure2.x_range = figure1.x_range
    figure3.x_range = figure1.x_range


    # stack_dataframes() has side-effects on the input, so we can only do it once
    accounts_stacked = stack_dataframes(accounts)

    add_balances_plot(figure1, accounts, accounts_stacked, annotations, analysis)
    figures_to_plot = [figure1]

    if analysis.has_capgains:
        add_invest_and_gains_plot(figure1, accounts, annotations, analysis)
        add_capital_returns_plot(figure2, accounts, annotations, analysis)
        figures_to_plot += [figure2]

    if analysis.daily_savings is not None:
        add_spending_and_savings_plot(figure3, annotations, analysis)
        figures_to_plot += [figure3]


    figure2.height = 300
    figure3.height = 300

    # Suppressing mypy error, not sure how to fix it...
    vertical_crosshair = bk.models.Span(dimension='height', line_dash='dotted', line_width=1)  # type: ignore[attr-defined]

    # Apply reasonable defaults for legends:
    for figure in figures_to_plot:
        figure.legend.location = 'top_left'
        figure.legend.click_policy = 'hide'

        figure.add_tools(bk.models.CrosshairTool(overlay=vertical_crosshair))

    # Note: gridplot got broken in 3.0.0 (still is in 3.0.1). It didn't draw
    # correctly. I switched to column() as a workaround, but it doesn't have
    # the merge_tools feature. TODO: Switch back as soon as possible.
    plot = bk.layouts.column(figures_to_plot, sizing_mode='stretch_both')
    bkp.output_file(file, title='ltfa ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    bkp.save(plot, resources=CustomResources(mode='inline'))


class CustomResources(bk.resources.Resources):
    @property
    def css_raw(self) -> list[str]:
        return super().css_raw + [
            """
            .bk-root .bk-tooltip>div {
                border-spacing: 2px;
            }
            .bk-root .bk-tooltip>div>div {
            }
            .bk-root .bk-tooltip>div>div:not(:first-child) .bk-tooltip-date {
                /* Add vertical space between several days */
                border-top: 1px solid lightgray;
                padding-top: 5px;
            }
            """
        ]
