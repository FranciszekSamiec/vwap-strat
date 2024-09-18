import yfinance as yf
import mplfinance as mpf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import ccxt 
import time
import plotly.graph_objs as go
import plotly.io as pio
import os





# BEGIN OF DATA PREPARATION ##################################################################

# this list tells the script not only which data to download but also which to trade
list_of_pairs = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'LTC/USDT', 'ADA/USDT',
                 'DOT/USDT', 'LINK/USDT', 'BNB/USDT', 'XLM/USDT', 'DOGE/USDT',\
                 'BCH/USDT', 'UNI/USDT', 'AAVE/USDT',\
                 'EOS/USDT', 'XMR/USDT', 'TRX/USDT', 'XTZ/USDT', 'VET/USDT',\
                 'ATOM/USDT']

# list_of_pairs = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'LTC/USDT', 'ADA/USDT']

# list_of_pairs = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'LTC/USDT', 'ADA/USDT',\
# 'DOT/USDT', 'LINK/USDT', 'BNB/USDT', 'XLM/USDT', 'DOGE/USDT', 'BCH/USDT']

# list_of_pairs = ['BTC/USDT']

# list_of_pairs = ['DOGE/USDT']




# Initialize the Binance exchange object
exchange = ccxt.binance({
    'rateLimit': 2000,
    'enableRateLimit': True,
})

timeframe = '1h'  # 1 hour candles


# bessa + bull run
#-----------------------------------
# days_back = 1000
# end_date = datetime(year=2024, month=7, day=30, hour=12, minute=0, second=0)
#-----------------------------------


# only bessa 
#-----------------------------------
days_back = 411
end_date = datetime(year=2022, month=12, day=30, hour=12, minute=0, second=0)
#-----------------------------------

start_date = end_date - timedelta(days=days_back) 




# Function to fetch OHLCV data in chunks
def fetch_ohlcv_in_chunks(exchange, symbol, timeframe, start_date, end_date, limit=1000):
    since = int(start_date.timestamp() * 1000)
    until = int(end_date.timestamp() * 1000)
    all_data = []

    while since < until:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            all_data += ohlcv
            # Update since to the timestamp of the last fetched candle +1 to avoid overlaps
            since = ohlcv[-1][0] + 1
            # Print progress
            print(f"Fetched {len(ohlcv)} candles. Next fetch from {datetime.utcfromtimestamp(since / 1000)}")
            # Sleep to respect the rate limit
            time.sleep(exchange.rateLimit / 1000)
        except ccxt.NetworkError as e:
            print(f"Network error: {str(e)}. Retrying in 10 seconds...")
            time.sleep(10)
        except ccxt.ExchangeError as e:
            print(f"Exchange error: {str(e)}. Retrying in 10 seconds...")
            time.sleep(10)
    
    return all_data


def vwap(df):
    typicalPrice = []
    cumVol = []
    if df.empty:
        print('Empty DataFrame, cannot calculate vwap for this pair.')
    else:
        cumVol.append(df.iloc[0, 4]) # volume

    cumTypPrice = []
    vwap = []


    for x in range(0, len(df.index)):
        typicalPrice.append((df.iloc[x, 1] + df.iloc[x, 2] + df.iloc[x, 3]) / 3)


    for x in range(0, len(df.index)):
        typicalPrice[x] = typicalPrice[x] * df.iloc[x, 4] 
        

    cumTypPrice.append(typicalPrice[0])

    vwap.append(cumTypPrice[0] / cumVol[0])

    for x in range(1, len(df.index)):
        if (df['timestamp'].iloc[x].hour == 0):
            cumVol.append(df.iloc[x, 4])
            cumTypPrice.append(typicalPrice[x])
        else:
            cumVol.append(cumVol[x-1] + df.iloc[x, 4])
            cumTypPrice.append(cumTypPrice[x - 1] + typicalPrice[x])
        vwap.append(cumTypPrice[x] / cumVol[x])

    return vwap


# Function to calculate the lowest low over a rolling window
def calculate_lowest_low(df, window):
    
    return df['low'].rolling(window=window).min()

def calculate_highest_high(df, window):

    return df['high'].rolling(window=window).max()


def prepare_data(exchange, list_of_pairs, timeframe, start_date, end_date):
        
    for pair in list_of_pairs:

        # Fetch historical OHLCV data in chunks
        ohlcv_data = fetch_ohlcv_in_chunks(exchange, pair, timeframe, start_date, end_date)

        # Convert to DataFrame and set
        #  column names
        ohlc_data = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Convert timestamp to datetime format
        ohlc_data['timestamp'] = pd.to_datetime(ohlc_data['timestamp'], unit='ms')

        # Filter the DataFrame to only include rows between the start_date and end_date
        ohlc_data = ohlc_data[(ohlc_data['timestamp'] >= start_date) & (ohlc_data['timestamp'] <= end_date)]

        # Calculate VWAP
        ohlc_data['VWAP'] = vwap(ohlc_data)

        # Define the rolling window size (e.g., 14 periods)
        window_size = 20

        # Add the lowest low column to ohlc_data
        ohlc_data['ll'] = calculate_lowest_low(ohlc_data, window_size)
        ohlc_data['hh'] = calculate_highest_high(ohlc_data, window_size)

        pair = pair.replace('/', '_')
        os.makedirs(pair, exist_ok=True)

        file_path = os.path.join(pair, f'{pair}_ohlcv_data.xlsx')

        ohlc_data.to_excel(file_path)


# prepare_data(exchange, list_of_pairs, timeframe, start_date, end_date)


# END OF DATA PREPARATION ################################################################













# BEGIN OF PLOTTING ####################################################################3

def add_trade_rectangles(fig, ohlc_data, entry_timestamps, risk_reward_ratio, list_of_durations):
    for timestamp, duration in zip(entry_timestamps, list_of_durations):
        # Get the row corresponding to the entry timestamp
        entry_row = ohlc_data.loc[ohlc_data['timestamp'] == timestamp]
        direction = entry_row['entry_condition'].values[0]
        if entry_row.empty:
            continue
        
        entry_price = entry_row['close'].values[0]
        
        if direction == 'long':
            stop_loss = entry_row['ll'].values[0]
            take_profit = entry_price + (entry_price - stop_loss) * risk_reward_ratio
        else:
            stop_loss = entry_row['hh'].values[0]
            take_profit = entry_price - (stop_loss - entry_price) * risk_reward_ratio
        

        # Define the end timestamp based on duration
        end_timestamp = timestamp + pd.Timedelta(hours=duration)

        # Find the exit candle within the duration
        exit_row = ohlc_data[(ohlc_data['timestamp'] >= timestamp) & (ohlc_data['timestamp'] <= end_timestamp)].iloc[-1]

        # Add the entry to take profit/duration rectangle
        fig.add_shape(type="rect",
                      x0=timestamp, y0=entry_price, x1=exit_row['timestamp'], y1=take_profit,
                      line=dict(color="Green", width=1),
                      fillcolor="LightGreen",
                      opacity=0.3,
                      layer="below")

        # Add the entry to stop loss rectangle
        fig.add_shape(type="rect",
                      x0=timestamp, y0=entry_price, x1=exit_row['timestamp'], y1=stop_loss,
                      line=dict(color="Red", width=1),
                      fillcolor="LightCoral",
                      opacity=0.3,
                      layer="below")


def plot_candlestick_with_highlights(ohlc_data, pair, rrr, list_of_durations):

    entry_timestamps = ohlc_data[ohlc_data['entry_condition'].notna()]['timestamp']

    # Create the candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=ohlc_data['timestamp'],
                                        open=ohlc_data['open'],
                                        high=ohlc_data['high'],
                                        low=ohlc_data['low'],
                                        close=ohlc_data['close'])])

    # Add VWAP line
    fig.add_trace(go.Scatter(x=ohlc_data['timestamp'], y=ohlc_data['VWAP'], mode='lines', name='VWAP', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=ohlc_data['timestamp'], y=ohlc_data['ll'], mode='lines', name='ll', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=ohlc_data['timestamp'], y=ohlc_data['hh'], mode='lines', name='hh', line=dict(color='green')))


    # Add markers for the highlight timestamps
    highlight_data = ohlc_data[ohlc_data['timestamp'].isin(entry_timestamps)]
    fig.add_trace(go.Scatter(x=highlight_data['timestamp'], y=highlight_data['high'] * 1.01, mode='markers', 
                            marker=dict(color='blue', size=10, symbol='cross'), name='Highlighted Candles'))


    # Add trade rectangles
    add_trade_rectangles(fig, ohlc_data, entry_timestamps, rrr, list_of_durations)

    # Update layout
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=False),  # Disable the range slider
            title='Timestamp',
            fixedrange=False  # Allows zooming and panning

        ),
        yaxis=dict(
            title='Price (USDT)',
            fixedrange=False  # Allows zooming and panning
        ),
        dragmode='zoom',
        # hovermode='x',
        title=f'{pair} rrr: {rrr} Hourly Candlestick Chart with Custom Highlights'
    )

    # Show the plot
    pio.show(fig)

# plot_candlestick_with_highlights(pd.read_excel('BTC_USDT/BTC_USDT_ohlcv_data.xlsx'))

# END OF PLOTTING #####################################################################













# BEGIN OF ENTRY CONDITION ############################################################### 

# Function to navigate through the DataFrame rows based on a timestamp
def get_relative_row(df, timestamp, offset):
    """
    Get a row from the DataFrame relative to a given timestamp.

    Parameters:
    - df: DataFrame, the input DataFrame with a datetime index.
    - timestamp: datetime, the specific timestamp to reference.
    - offset: int, how many rows before or after the timestamp to retrieve.

    Returns:
    - The row at the specified relative position.
    """
    try:
        pos = df.index.get_loc(timestamp)
        relative_pos = pos + offset
        if 0 <= relative_pos < len(df):
            return df.iloc[relative_pos]
        else:
            return None
    except KeyError:
        print(f"Timestamp {timestamp} not found in the DataFrame.")
        return None


def direction_comparator(row_a, row_b, direction, row_a_col_long, row_b_col_long, row_a_col_short, row_b_col_short):
    # > is for long,
    # < is for short

    if direction == 'long':
        res = row_a[row_a_col_long] > row_b[row_b_col_long]
    else:
        res = row_a[row_a_col_short] < row_b[row_b_col_short]
    # else:   
    #     if direction == 'long':
    #         res = row_a['high'] > row_b['VWAP']
    #     else:   
    #         res = row_a['low'] < row_b['VWAP']
            
    return res
    

def cond_logic(direction, hour, row, previous_row, prev_prev, prev_3x):
    if hour == 1:
        # basic shape of vwap recalculation is met
        #-----------------------------------------
        if direction_comparator(previous_row, prev_prev, direction, 'VWAP', 'VWAP', 'VWAP', 'VWAP'):
            return False 
        #-----------------------------------------
        
        # wicks do not overlap vwap from recalculation point
        #---------------------------------------------------
        if direction_comparator(row, prev_prev, direction, 'high', 'VWAP', 'low', 'VWAP'):
            return False
        # if row['high'] > prev_prev['VWAP'] :
        #     return False
        if direction_comparator(previous_row, prev_prev, direction, 'high', 'VWAP', 'low', 'VWAP'):
            return False
        # if previous_row['high'] > prev_prev['VWAP']:
        #     return False
        #---------------------------------------------------

        # candles are sufficiently below vwap before recalc
        #--------------------------------------------------
        if direction_comparator(prev_prev, prev_prev, direction, 'high', 'VWAP', 'low', 'VWAP'):
            return False
        # if prev_prev['high'] > prev_prev['VWAP']:
        #     return False
        # if prev_3x['high'] > prev_prev['VWAP']:
        #     return False
        # if prev_4x['high'] > prev_prev['VWAP']:
        #     return False
        #--------------------------------------------------
        
    if hour == 2:
        # print(row['timestamp'])
        # print(direction)
        # basic shape of vwap recalculation is met
        #-----------------------------------------
        if direction_comparator(prev_prev, prev_3x, direction, 'VWAP', 'VWAP', 'VWAP', 'VWAP'):
            return False
        # if prev_prev['VWAP'] > prev_3x['VWAP']:
        #     return False
        #-----------------------------------------

        # wicks do not overlap vwap from recalculation point
        #---------------------------------------------------
        if direction_comparator(row, prev_3x, direction, 'high', 'VWAP', 'low', 'VWAP'):
            return False
        # if row['high'] > prev_3x['VWAP']:
        #     return False
        if direction_comparator(previous_row, prev_3x, direction, 'high', 'VWAP', 'low', 'VWAP'):
            return False
        # if previous_row['high'] > prev_3x['VWAP']:
        #     return False
        if direction_comparator(prev_prev, prev_3x, direction, 'high', 'VWAP', 'low', 'VWAP'):
            return False
        # if prev_prev['high'] > prev_3x['VWAP']:
        #     return False
        #---------------------------------------------------

        # candles are sufficiently below vwap before recalc
        #--------------------------------------------------
        if direction_comparator(prev_3x, prev_3x, direction, 'high', 'VWAP', 'low', 'VWAP'):
            return False
        # if prev_3x['high'] > prev_3x['VWAP']:
        #     return False
        # if prev_4x['high'] > prev_3x['VWAP']:
        #     return False
        # if prev_5x['high'] > prev_3x['VWAP']:
        #     return False
        #--------------------------------------------------

    # print(row['timestamp'])
    # Check if the price crosses the vwap recalc point in a favorable direction
    return  direction_comparator(row, row, direction, 'close', 'VWAP', 'close', 'VWAP') and \
            direction_comparator(previous_row, previous_row, direction, 'VWAP', 'close', 'VWAP', 'close') and \
            direction_comparator(row, row, direction, 'close', 'open', 'close', 'open')
    


def entry_cond(ohlc_data, idx, direction):

    # print(ohlc_data)
    # specific_timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    timestamp = ohlc_data['timestamp'].iloc[idx]
    hour = timestamp.hour
    if hour not in [1, 2]:
        return False
    
    row = ohlc_data.iloc[idx]


    if idx <= 5 or idx >= len(ohlc_data) - 6:
        return False
    
    if pd.isna(row['ll']):
        # print('No ll value')
        return False
    

    previous_row = ohlc_data.iloc[idx - 1]
    prev_prev = ohlc_data.iloc[idx - 2]
    prev_3x = ohlc_data.iloc[idx - 3]
    prev_4x = ohlc_data.iloc[idx - 4]
    prev_5x = ohlc_data.iloc[idx - 5]
    prev_6x = ohlc_data.iloc[idx - 6]


    return cond_logic(direction, hour, row, previous_row, prev_prev, prev_3x)



def skip_position_duration(ohlc_data, entry_idx, take_profit, stop_loss, direction, duration_hours=24):

    entry_timestamp = ohlc_data.iloc[entry_idx]['timestamp'] 
    for i in range(entry_idx + 1, len(ohlc_data)):
        row = ohlc_data.iloc[i]
        high = row['high']
        low = row['low']

        # Check if the stop loss is hit
        if direction == 'long':
            if low <= stop_loss:
                result = "SL Hit"
                break

            # Check if the take profit is hit
            if high >= take_profit:
                result = "TP Hit"
                break
        else:
            if high >= stop_loss:
                result = "SL Hit"
                break

            # Check if the take profit is hit
            if low <= take_profit:
                result = "TP Hit"
                break

        # Check if the duration has ended - alternative approach could be that no end of duration is defined
        # for alternative approach comment out the following block:
        # -------------------------------------------------------------------------------------------
        if row['timestamp'] >= entry_timestamp + pd.Timedelta(hours=duration_hours):
            result = "Duration Ended"
            break
        # -------------------------------------------------------------------------------------------

    return i, i - entry_idx


def add_entry_cond(ohlc_data, pair, rrr, plot = False, long=True, short=True):

    # Applying the entry condition function to each timestamp
    # ohlc_data['entry_condition'] = ohlc_data.index.to_series().apply(lambda x: entry_cond(ohlc_data, x))

    # alternative approach - trades are not allowed to overlap

    ohlc_data['entry_condition'] = None

    list_of_durations = []
    i = 0
    while i < len(ohlc_data):
        row = ohlc_data.iloc[i]
        if long and entry_cond(ohlc_data, i, 'long'):
            ohlc_data.at[i, 'entry_condition'] = 'long'
            sl = row['ll']
            tp = row['close'] + (row['close'] - sl) * rrr
            i, duration = skip_position_duration(ohlc_data, i, tp, sl, "long")
            list_of_durations.append(duration)
        elif short and entry_cond(ohlc_data, i, 'short'):
            ohlc_data.at[i, 'entry_condition'] = 'short'
            sl = row['hh']
            tp = row['close'] - (sl - row['close']) * rrr
            i, duration = skip_position_duration(ohlc_data, i, tp, sl, "short")
            list_of_durations.append(duration)
        else:
            ohlc_data.at[i, 'entry_condition'] = None
            i += 1
        
    if plot:
        plot_candlestick_with_highlights(ohlc_data, pair, rrr, list_of_durations)

    # Extracting the timestamps where the condition is True
    entry_timestamps = ohlc_data[ohlc_data['entry_condition'].notna()].index.tolist()
    # print(entry_timestamps)
    # # print(entry_prices)

    return entry_timestamps

# END OF ENTRY CONDITION ###############################################################












# BEGIN OF positions handling #########################################################

def execute_position(df, entry_timestamp, direction, risk_reward_ratio, duration_hours, initial_capital, risk_percentage, trading_fee):
    """
    Executes a simulated long position on historical data with risk management.

    Parameters:
    - df: DataFrame, the input DataFrame containing OHLCV data with a 'timestamp' column.
    - entry_timestamp: datetime, the timestamp to enter the position.
    - stop_loss: float, the stop loss price.
    - risk_reward_ratio: float, the risk-reward ratio.
    - duration_hours: int, the duration of the position in hours.
    - initial_capital: float, the initial capital for the trade.
    - risk_percentage: float, the percentage of initial capital to risk per trade.

    Returns:
    - result: str, the result of the trade ('TP Hit', 'SL Hit', 'Duration Ended', 'Not Enough Data').
    - final_capital: float, the capital after the trade.
    - risked_amount: float, the amount of capital risked based on stop loss.
    """

    # Find the index of the entry candle
    entry_idx = df[df['timestamp'] == entry_timestamp].index[0]

    # Check if the entry_timestamp is in the DataFrame
    if entry_timestamp not in df['timestamp'].values:
        return "Entry timestamp not in data.", initial_capital, 0

    # Calculate the entry price
    entry_price = df.at[entry_idx, 'close']

    if direction == 'long':
        stop_loss = df.at[entry_idx, 'll']
        take_profit = entry_price + (entry_price - stop_loss) * risk_reward_ratio
        # Calculate the amount to risk per unit (difference between entry price and stop loss)
        risk_per_unit = entry_price - stop_loss

        sl_hit_cond = lambda row: row['low'] <= stop_loss
        tp_hit_cond = lambda row: row['high'] >= take_profit
    else:
        stop_loss = df.at[entry_idx, 'hh']
        take_profit = entry_price + (entry_price - stop_loss) * risk_reward_ratio
        risk_per_unit = stop_loss - entry_price

        sl_hit_cond = lambda row: row['high'] >= stop_loss
        tp_hit_cond = lambda row: row['low'] <= take_profit


    # Calculate the maximum amount of capital to risk
    max_risk = initial_capital * (risk_percentage / 100)



    if risk_per_unit <= 0:
        print("Invalid stop loss (below entry price).")
        return "Invalid stop loss (below entry price).", initial_capital, 0
    # print(direction)

    # Calculate the position size (number of units)
    position_size = max_risk / risk_per_unit

    # Calculate the total amount of capital needed for the position
    total_position_cost = position_size * entry_price

    # Adjust the position size if it exceeds the available capital
    if total_position_cost > initial_capital:
        position_size = initial_capital / entry_price
        total_position_cost = initial_capital
        max_risk = position_size * risk_per_unit


    # Define the end of the duration period
    end_timestamp = entry_timestamp + timedelta(hours=duration_hours)
    if end_timestamp > df['timestamp'].iloc[-1]:
        return "Not enough data to complete the trade duration.", initial_capital, 0

    # Initialize variables for trade result
    result = "Duration Ended"
    final_price = df['close'].iloc[-1]


    # Iterate through the candles starting from the entry point
    for i in range(entry_idx + 1, len(df)):
        row = df.iloc[i]

        # Check if the stop loss is hit
        if sl_hit_cond(row):
            result = "SL Hit"
            final_price = stop_loss
            break

        # Check if the take profit is hit
        if tp_hit_cond(row):
            result = "TP Hit"
            final_price = take_profit
            break

        # Check if the duration has ended
        if row['timestamp'] >= end_timestamp:
            result = "Duration Ended"
            final_price = row['close']
            break


    # Calculate fees on the entry and exit transactions
    entry_fee = total_position_cost * trading_fee
    exit_fee = position_size * final_price * trading_fee
    total_fees = entry_fee + exit_fee

    # Calculate final capital after accounting for fees
    final_capital = initial_capital
    if result == "TP Hit":
        final_capital += position_size * abs(take_profit - entry_price) - total_fees
    elif result == "SL Hit":
        final_capital -= max_risk + total_fees 
    else:  # Duration Ended
        # print(direction)
        if direction == 'long':
            final_capital += position_size * (final_price - entry_price) - total_fees
        elif direction == 'short':
            
            final_capital += position_size * (entry_price - final_price) - total_fees

    return result, final_capital, total_fees

# END OF position handling #############################################################













# BEGIN OF trading #####################################################################

# timestamp needs to be an index in dataframe when passing to add entry cond
# but it needs to be restored to a column when passing to execute_position


# Initial capital
initial_capital = 100000  # Example initial capital
risk_percentage = 2       # Risk 2% of initial capital per trade
duration_hours = 24       # Example duration of 24 hours for each trade
trading_fee = 0.001       # 0.1% trading fee
longs = True              # Allow long trades
shorts = True            # Allow short trades
enforce_max_duration = True  # Enforce maximum duration for trades

# final results will store list of tuples (pair, rrr, capital) capital is the 
# final capital after all trades for the pair and rrr
final_results = [] 


def read_pair(pair):    
    pair = pair.replace('/', '_')
    ohlc_data = pd.read_excel(f'{pair}/{pair}_ohlcv_data.xlsx')
    # ohlc_data = ohlc_data.reset_index()

    return ohlc_data


# Function to save capital progression plot
def save_capital_plot(capital_list, rrr, pair):
    plt.figure(figsize=(12, 6))
    plt.plot(capital_list, marker='o', linestyle='-', color='b', markersize=5)
    plt.title(f'Capital Progression Over Trades (RRR = {rrr})')
    plt.xlabel('Trade Number')
    plt.ylabel('Capital (USDT)')
    plt.grid(True)

    pair = pair.replace('/', '_')
    # Save the capital progression plot for the current RRR
    os.makedirs(pair, exist_ok=True)
    file_path = os.path.join(pair, f'capital_progression_rrr_{rrr}.png')

    plt.savefig(file_path)
    plt.close()



for pair in list_of_pairs:

    ohlc_data = read_pair(pair)

    # Loop through risk-reward ratios from 1 to 10
    for rrr in range(1, 11):

        if rrr == 4 and (pair == 'BTC/USDT' or pair == 'DOGE/USDT'):
            add_entry_cond(ohlc_data, pair, rrr, plot=True, long=longs, short=shorts)
        else:
            add_entry_cond(ohlc_data, pair, rrr, plot=False, long=longs, short=shorts)

        # List to track capital over time for current RRR
        capital_list = [initial_capital]

        for index, row in ohlc_data[ohlc_data['entry_condition'].notna()].iterrows():
            entry_timestamp = row['timestamp']

            
            # Execute the position
            result, final_capital, fees = execute_position(
                ohlc_data, entry_timestamp, row['entry_condition'], rrr, duration_hours, capital_list[-1], risk_percentage, trading_fee
            )


            # Append the final capital after this trade to the list
            capital_list.append(final_capital)

        final_results.append((pair, rrr, capital_list[-1]))
        save_capital_plot(capital_list, rrr, pair)


# BEGIN OF EVALUATION METRICS------------------------------------------------------------

pair_to_highest_capital = {}

# Process the list to populate the dictionary
for pair, rrr, capital in final_results:
    if pair not in pair_to_highest_capital:
        # If the pair is not in the dictionary, add it
        pair_to_highest_capital[pair] = (capital, rrr)
    else:
        # If the pair is already in the dictionary, check if the new capital is higher
        current_highest_capital, current_best_rrr = pair_to_highest_capital[pair]
        if capital > current_highest_capital:
            pair_to_highest_capital[pair] = (capital, rrr)

pair_to_highest_capital = {
    pair: (float(capital), rrr)
    for pair, (capital, rrr) in pair_to_highest_capital.items()
}

# convert to percentages
def convert_to_percentage(capital, initial_capital):
    fract = (capital - initial_capital) / initial_capital
    perc = round(fract * 100, 2)
    return perc

pair_to_highest_capital = {
    pair: (convert_to_percentage(capital, initial_capital), rrr)
    for pair, (capital, rrr) in pair_to_highest_capital.items()
}

pair_to_highest_capital = dict(sorted(
    pair_to_highest_capital.items(),
    key=lambda item: item[1][0],  # Sort by the first element in the tuple (capital)
    reverse=True  # Sort in descending order
))

# Print the summary of simulation
print('start date:', start_date)
print('end date:', end_date)
print('longs:', longs)
print('shorts:', shorts)
print('enforce max duration:', enforce_max_duration)
if enforce_max_duration:
    print('max duration:', duration_hours, ' hours')

for pair, (capital, rrr) in pair_to_highest_capital.items():
    print(f"Pair: {pair}, Capital gain (%): {capital:.2f}, RRR: {rrr}")

# END OF EVALUATION METRICS------------------------------------------------------------

# END OF trading #######################################################################
    




