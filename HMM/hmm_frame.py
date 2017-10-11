from HMM import Bar

train_file_name = 'rb88_daily.txt'
test_file_name = 'rb1801_daily.txt'


def load_bars(file_name):
    bars = []
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            strs = line.split(',')
            strs = list(map(lambda x: x.strip(), strs))
            date = strs[0]
            time = strs[1]
            open_value = float(strs[2])
            high_value = float(strs[3])
            low_value = float(strs[4])
            close_value = float(strs[5])
            bar = Bar(date, time, open_value, high_value, low_value, close_value)
            bars.append(bar)
    return bars


train_bars = load_bars(train_file_name)
test_bars = load_bars(test_file_name)

def transform_bar_to_frame_data(bars, pre=16, post=1):
    frame_data = []
    pass
