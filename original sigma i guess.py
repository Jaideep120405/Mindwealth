# sigma(price) indicators.

''' check for the daily returns... and arrange in increasing order. and plot the distribution and check for the sigma moves. 
    plot the price and below the weekly returns. 
    highlight the points where there are weekly sigma moves(>1 sigma).

'''

from dash import dcc, callback, Output, Input, State
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas as pd
from constant import *
from compute import *
from config import *
from yahoo import *
from plot import *
from util import *
from data import *
from ui import *
import plotly.graph_objects as go
import dash
import os
import matplotlib.pyplot as plt

dash.register_page(__name__, path='/sigma', name='Sigma', order='19')

scenario_div = get_scenario_div([
    get_symbol_input(),
])
parameter_div = get_parameter_div([
    get_analyze_button('sigma'),
])

out_tab = get_out_tab({
    'Plot': get_plot_div(),
    'Report': get_report_div()
})
layout = get_page_layout('Sigma', scenario_div,parameter_div , out_tab)

# Triggered when Analyze button clicked
@callback(
    [
		Output('alert-dlg', 'is_open', allow_duplicate = True),
		Output('alert-msg', 'children', allow_duplicate = True),
		Output('alert-dlg', 'style', allow_duplicate = True),
		Output('out-plot', 'children', allow_duplicate = True),
        Output('out-report', 'children', allow_duplicate = True)
    ],
    Input('sigma-analyze-button', 'n_clicks'),
    [
        State('symbol-input', 'value'),
    ],
    prevent_initial_call=True
)

def on_analyze_clicked(n_clicks, symbol):
    none_ret, traces = [None], []  # Padding return values
    if n_clicks == 0:
        return alert_hide(none_ret)
    if symbol is None:
        return alert_error('Invalid symbol. Please select one and retry.', none_ret)

    ipo_date = load_stake().loc[symbol]['ipo']
    from_date=ipo_date
    to_date=datetime.today()
    # Read the data
    end_date = datetime.today().date() + timedelta(days=1)
    start_date = end_date - timedelta(days=30)  # 1 month ago

    # Fetch historical stock data
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    # Extract the closing prices
    closing_prices = stock_data['Close']

    # If end_date is not a trading day, consider the previous trading day
    if end_date not in closing_prices.index:
        end_date = closing_prices.index[-1]

    # If start_date is not a trading day, consider the previous tradingday
    if start_date not in closing_prices.index:
        start_date = closing_prices.index[0]

    # Get the closing prices for start_date and end_date
    start_price = closing_prices.loc[start_date]
    end_price = closing_prices.loc[end_date]

    # Calculate the percentage change
    live_percentage_change = ((end_price - start_price) / start_price) * 100
    # Print the result
   
    df = download_monthly_ticker_data(symbol, from_date, to_date)

    # Calculate monthly returns and percentage returns
    df['monthly_returns'] = df['Close'] - df['Open']
    df['monthly_returns_perc'] = df['monthly_returns'] * 100 / df['Close']
    # Create a histogram
    jd = df['monthly_returns_perc'].sort_values()
    average_monthly_returns_perc = np.mean(df['monthly_returns_perc'])
    std_dev_monthly_returns_perc = np.std(df['monthly_returns_perc'])
    sigma_level_live = (live_percentage_change-average_monthly_returns_perc)/std_dev_monthly_returns_perc
    
    df['sigma_level'] = (df['monthly_returns'] - average_monthly_returns_perc)/std_dev_monthly_returns_perc
    figure = make_subplots(rows = 2, cols = 1, shared_xaxes = False, vertical_spacing = 0.2, subplot_titles = ('Distribution', 'Instances'),row_width = [0.7,0.3])

    figure.add_trace(go.Histogram(x=jd, nbinsx=50),row=1,col=1)
    # Specify the value of x you want to indicate
    value_of_x = 0
    
    # Add a vertical line at the specified x value
    draw_vline_shape(figure, value_of_x,0,100 , color = 'red', width = 1.5, dash = 'solid',row=1,col=1)
    
    
    # Annotate the line with a small triangle pointer
    figure.add_annotation(
        x=value_of_x,
        y=50,
        text=f'x = {value_of_x}',
        showarrow=True,
        arrowhead=1,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor="black",
        ax=20,
        ay=-30,
        font=dict(color="black", size=10),
        row=1,col=1
    )
    
    value_of_x = live_percentage_change
    draw_vline_shape(figure, value_of_x,0,100 , color = 'black', width = 0.5, dash = 'solid',row=1,col=1)
    sigma=value_of_x
    
    # Annotate the line with a small triangle pointer
    figure.add_annotation(
        x=value_of_x,
        y=80,
        text=f'x = {value_of_x}',
        showarrow=True,
        arrowhead=1,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor="black",
        ax=20,
        ay=-30,
        font=dict(color="black", size=10),
        row=1,col=1
    )
    
    
    # Select data points within a range
    if live_percentage_change >=0:
        history = df[df['monthly_returns_perc'].between(live_percentage_change,1000)].index
    else:
        history = df[df['monthly_returns_perc'].between(-1000,live_percentage_change)].index

    zero_sigma = df[df['monthly_returns_perc'].between(average_monthly_returns_perc-0.1*std_dev_monthly_returns_perc,
                                                    average_monthly_returns_perc+0.1*std_dev_monthly_returns_perc)].index

    # Plot the Close prices over time
    #fig2 = go.Figure()
    figure.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines',line = dict(color ='red')),row=2,col=1)
    
    # Add vertical lines for historical data
    for i in history:
        value_of_x = df['Date'].iloc[i]
    
        figure.add_shape(
            dict(type="line", x0=value_of_x, y0=min(df['Close']), x1=value_of_x, y1=max(df['Close']), line=dict(color="black", width=1.5)),row=2,col=1
        )
    
        figure.add_annotation(
            x=value_of_x,
            y=min(df['Close']),
            text=f'x = {value_of_x}',
            showarrow=True,
            arrowhead=1,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="blue",
            ax=20,
            ay=-30,
            textangle=90,
            font=dict(color="black", size=8),
            row=2,col=1
        )

    figure.update_layout(
    height=900,  # Set the height in pixels
    width=1300,   # Set the width in pixels
    )
    figure.update_xaxes(title_text='%month-month change', row=1, col=1)
    figure.update_xaxes(title_text='Date', row=2, col=1)
    
    # Set y-axis titles
    figure.update_yaxes(title_text='Frequency(months)', row=1, col=1)
    figure.update_yaxes(title_text='Price', row=2, col=1)

    one_month = []
    onee_month = []
    three_month = []
    six_month = []
    twelve_month = []
    date = []
    sigma_level = []


    for i in history:
        value_of_x = df['Date'][i]
        date.append(df['Date'][i])

        try:
            sigma_level.append(((100*(df['Close'][i]-df['Close'][i-1])/df['Close'][i-1])-average_monthly_returns_perc)/std_dev_monthly_returns_perc)
        except Exception as e:
            sigma_level.append("NA")

        try:
            one_month.append(100*(df['Close'][i]-df['Close'][i-1])/df['Close'][i-1])
        except Exception as e:
            one_month.append("NA")

        try:
            onee_month.append(100*(df['Close'][i+1]-df['Close'][i])/df['Close'][i])
        except Exception as e:
            onee_month.append("NA")
        
        try:
            three_month.append(100*(df['Close'][i+3]-df['Close'][i])/df['Close'][i])
        except Exception as e:
            three_month.append("NA")

        try:
            six_month.append(100*(df['Close'][i+6]-df['Close'][i])/df['Close'][i])
        except Exception as e:
            six_month.append("NA")

        try:
            twelve_month.append(100*(df['Close'][i+12]-df['Close'][i])/df['Close'][i])
        except Exception as e:
            twelve_month.append("NA")

        #print(df['sigma_level'].describe())
    print(symbol)
    print(f"1-month return from {start_date} to {end_date}: {live_percentage_change:.2f}%")
    number=sigma_level_live
    number2=len(history)
    
    IPO_date=f"IPO date for {symbol}:{from_date} . Today,we are at {number} sigma level.There were total {number2} instances of more than {number} sigma move in history"
    symbol={"Ticker":[symbol]}
    symm=pd.DataFrame(symbol)
    statement={f"Noting all cases where deviation is greater than {number} sigma"}
    statement2=pd.DataFrame(statement)
    data = {'past-month':one_month,'1-month': onee_month,'3-month': three_month, '6-month': six_month, '12-month': twelve_month, 'date' : date,'sigma': sigma_level}
    df2 = pd.DataFrame(data)
    df2.to_csv('history.csv')
    df3 = pd.read_csv('history.csv')
    df4 = df3.dropna(subset=['past-month'])
    print(df4.drop('Unnamed: 0',axis=1))
    print(df4['3-month'].describe())
    ans = df4['3-month'].mean()
    
    symbol_html = html.P(f"Symbol: {symbol}")
    return_html = []
    return_html.append(symbol_html)

    return_html.append(html.P(f"1-month return from {start_date} to {end_date}: {live_percentage_change:.2f}%"))

    IPO_date = f"IPO date for {symbol}: {from_date}. Today, we are at {number:.2f} sigma level. There were total {number2} instances of more than {number:.2f} sigma move in history"
    return_html.append(html.P(IPO_date))

    statement_html = html.P("Noting all cases where deviation is greater than one sigma")
    return_html.append(statement_html)

    df4_html = html.Table([
        html.Tr([html.Th(col) for col in df4.columns])] +
        [html.Tr([html.Td("{:.2f}".format(df4.iloc[i][col]) if isinstance(df4.iloc[i][col], (int, float)) else df4.iloc[i][col]) for col in df4.columns]) for i in range(len(df4))])


    return_html.append(html.P(df4_html))

    stats_mean_html = html.P(f"Mean of '3-month' column: {df4['3-month'].describe()['mean']:.2f}")
    return_html.append(stats_mean_html)

    excel_file_path = 'out/stats_sigma.xlsx'
    with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
        symm.to_excel(writer, sheet_name='History', index=False, header=True, startrow=0, startcol=0)
        df2.to_excel(writer, sheet_name='History', index=False, header=True, startrow=symm.shape[0] + 2, startcol=0)
        statement2.to_excel(writer, sheet_name='Stats', index=False, header=True, startrow=0, startcol=0)
        df4.to_excel(writer, sheet_name='Stats', index=False, header=True, startrow=statement2.shape[0] + 2, startcol=0)
    #percentile:
    sorted_returns = np.sort(three_month)

    # Find the index positions of the largest negative value and smallest positive value
    max_neg_index = np.searchsorted(sorted_returns, 0, side='right')
    min_pos_index = np.searchsorted(sorted_returns, 0, side='left')

    # Interpolate the percentile based on the positions of the values around zero
    if max_neg_index == 0:
        percentile = 0
    elif min_pos_index == len(sorted_returns):
        percentile = 100
    else:
        percentile = (max_neg_index / len(sorted_returns)) * 100

    print("Percentile of zero:", f'{percentile:.1f}')
    print('average 1 month returns: ',f'{average_monthly_returns_perc:.2f}')
    print('Normalized_average_returns_in_3_months: ', f'{ans:.2f}',' - 3*',f'{average_monthly_returns_perc:.2f}',' = ',f'{(ans-3*average_monthly_returns_perc):.2f}')

    one_month = []
    onee_month = []
    three_month = []
    six_month = []
    twelve_month = []
    date = []
    sigma_level = []


    for i in zero_sigma:
        value_of_x = df['Date'][i]
        date.append(df['Date'][i])

        try:
            sigma_level.append(((100*(df['Close'][i]-df['Close'][i-1])/df['Close'][i-1])-average_monthly_returns_perc)/std_dev_monthly_returns_perc)
        except Exception as e:
            sigma_level.append("NA")

        try:
            one_month.append(100*(df['Close'][i]-df['Close'][i-1])/df['Close'][i-1])
        except Exception as e:
            one_month.append("NA")

        try:
            onee_month.append(100*(df['Close'][i+1]-df['Close'][i])/df['Close'][i])
        except Exception as e:
            onee_month.append("NA")
        
        try:
            three_month.append(100*(df['Close'][i+3]-df['Close'][i])/df['Close'][i])
        except Exception as e:
            three_month.append("NA")

        try:
            six_month.append(100*(df['Close'][i+6]-df['Close'][i])/df['Close'][i])
        except Exception as e:
            six_month.append("NA")

        try:
            twelve_month.append(100*(df['Close'][i+12]-df['Close'][i])/df['Close'][i])
        except Exception as e:
            twelve_month.append("NA")


    data = {'past-month':one_month,'1-month': onee_month,'3-month': three_month, '6-month': six_month, '12-month': twelve_month, 'date' : date,'sigma': sigma_level}
    df2 = pd.DataFrame(data)
    df2.to_csv('zero_sigma.csv')
    df3 = pd.read_csv('zero_sigma.csv')
    df4 = df3.dropna(subset=['past-month'])

    try:
        print('mean of 3-month returns after a zero sigma instance: ', f'{np.mean(three_month):.2f}')
        print('Normalized_average_returns_in_3_months: ', f'{ans:.2f}',' - ',f'{np.mean(three_month)}',' = ',f'{(ans-np.mean(three_month)):.2f}')
    except Exception as e:
        print(' ')
    
    # New printed information
    sorted_returns = np.sort(three_month)
    max_neg_index = np.searchsorted(sorted_returns, 0, side='right')
    min_pos_index = np.searchsorted(sorted_returns, 0, side='left')
    if max_neg_index == 0:
        percentile = 0
    elif min_pos_index == len(sorted_returns):
        percentile = 100
    else:
        percentile = (max_neg_index / len(sorted_returns)) * 100

    percentile_html = html.P(f"Percentile of zero: {percentile:.1f}")

    average_1_month_html = html.P(f"Average 1-month returns: {average_monthly_returns_perc:.2f}")
    
    normalized_returns_html = html.P(f"Normalized average returns in 3 months: {ans:.2f} - {np.mean(three_month)} = {(ans - np.mean(three_month)):.2f}")

    mean_3_month_html = html.P(f"Mean of 3-month returns after a zero sigma instance: {np.mean(three_month):.2f}")

    return_html.append(percentile_html)
    return_html.append(average_1_month_html)
    return_html.append(normalized_returns_html)
    return_html.append(mean_3_month_html)

    return alert_success('Analysis Completed') + [dcc.Graph(figure=figure)] + [html.Div(return_html)]

def download_monthly_ticker_data(ticker, start_date, end_date):
    # Download historical data
    data = yf.download(ticker, start=start_date, end=end_date, interval='1mo')

    # Reset the index to make 'Date' a column
    data.reset_index(inplace=True)

    # Format 'Date' column to mm-dd-yyyy
    data['Date'] = data['Date'].dt.strftime('%d-%m-%Y')

    return data