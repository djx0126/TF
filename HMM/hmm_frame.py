import math
import numpy as np
from matplotlib import cm, pyplot as plt
import pandas as pd

from HMM.Bar import Bar
from hmmlearn.hmm import MultinomialHMM

train_file_name = 'rb88_daily.txt'
verify_file_name = 'rb1801_daily.txt'

frame_length = 16
n_components = 5
n_value_seg = 5
n_iter = 500
ratio_for_test = 0.75

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
verify_bars = load_bars(verify_file_name)

train_close_seq = list(map(lambda x: x.close, train_bars))
train_date_seq = list(map(lambda x: x.date, train_bars))
verify_close_seq = list(map(lambda x: x.close, verify_bars))


def build_close_diff_seq(seq):
    diff_seq = []
    for i in range(len(seq)):
        if i > 0:
            diff_value = seq[i] - seq[i - 1]
            diff_value_ratio = diff_value / seq[i - 1]
            log_diff_value_ratio = math.log(diff_value_ratio + 1.0)
            diff_seq.append(log_diff_value_ratio)
    return diff_seq


train_diff = build_close_diff_seq(train_close_seq)
verify_diff = build_close_diff_seq(verify_close_seq)


def calc_sort_edge(value_array):
    n_each_seg = int(len(value_array) / n_value_seg)
    sorted_value = sorted(value_array)
    seg_edge_idx = []
    for i in range(n_value_seg):
        if i > 0:
            seg_edge_idx.append(n_each_seg * i)
    seg_edge = np.array(sorted_value)[seg_edge_idx]
    return seg_edge


seg_edge = calc_sort_edge(train_diff)
print(seg_edge)


train_len = int(ratio_for_test * len(train_diff))
test_diff = train_diff[train_len:]
train_diff = train_diff[:train_len]

print(train_diff[-5:])


def transform_to_int(value_array, seg_edge):
    new_array = []
    for value in value_array:
        if value >= seg_edge[len(seg_edge) - 1]:
            new_array.append(len(seg_edge))
        else:
            new_array.append(np.where(seg_edge > value)[0][0])
    return new_array


train_diff_int = transform_to_int(train_diff, seg_edge)
test_diff_int = transform_to_int(test_diff, seg_edge)
verify_diff_int = transform_to_int(verify_diff, seg_edge)
print(train_diff_int[-5:])

# split to frames



def transform_to_frames(value_array, frame_length):
    length = len(value_array)
    X_frame = []
    Y_frame = []
    for i in range(length - frame_length):  # reserve one for Y
        X_frame.append(value_array[i:i + frame_length + 1])
        Y_frame.append(value_array[i + frame_length:i + frame_length + 1])
    return X_frame, Y_frame


X_train, Y_train = transform_to_frames(train_diff_int, frame_length)
X_test, Y_test = transform_to_frames(test_diff_int, frame_length)
X_verify, Y_verify = transform_to_frames(verify_diff_int, frame_length)
print(X_train[-3:])
print(Y_train[-3:])
print('size X_train = ' + str(len(X_train)))
print('size X_test = ' + str(len(X_test)))
print('size X_verify = ' + str(len(X_verify)))


# transform to array for HMM
def transform_X_for_hmm(X):
    X_transformed = np.array(X[0]).reshape(-1, 1)
    X_lengths = [len(X[0])]
    for i in range(1, len(X)):
        X_transformed = np.concatenate([X_transformed, np.array(X[i]).reshape(-1, 1)])
        X_lengths.append(len(X[i]))
    return X_transformed, X_lengths


# print(X_train_hmm[-5:])
# print(X_train_hmm.shape)
# print(sum(X_train_lengths))




def predict_next(test_seq, clf):
    probs = []
    for seg in range(n_value_seg):
        testing_seq = np.array(test_seq)
        testing_seq[len(testing_seq) - 1] = seg
        prob, states = clf.decode(testing_seq.reshape(-1, 1))
        probs.append(prob)
        # print(prob)
    predict_value = probs.index(max(probs))
    return predict_value


def train_on_X(X):
    X_train_hmm, X_train_lengths = transform_X_for_hmm(X)
    clf = MultinomialHMM(n_components=n_components, n_iter=n_iter)
    clf.fit(X_train_hmm, lengths=X_train_lengths)
    return clf

def run_test_on_X(X, clf):
    right = 0
    loose_count = 0
    loose_rigth = 0
    for test_seq in X:
        expected = test_seq[len(test_seq) - 1]
        predict = predict_next(test_seq, clf)
        if predict == expected:
            right += 1
        if predict == n_value_seg-1:
            loose_count += 1
            if expected == n_value_seg-1 or expected == n_value_seg-2:
                loose_rigth += 1
        if predict == 0:
            loose_count += 1
            if expected == 0 or expected == 1:
                loose_rigth += 1

    accuracy = right / len(X)
    loose_accuracy = 0.4 if loose_count == 0 else loose_rigth / loose_count
    return accuracy, loose_accuracy


sum_accuracy = 0
sum_test_accuracy = 0
sum_verify_accuracy = 0
sum_loose_accuracy = 0
sum_loose_test_accuracy = 0
sum_loose_verify_accuracy = 0
n_run = 10
best_accuracy = 0

for i in range(n_run):
    clf = train_on_X(X_train)
    accuracy, loose_accuracy = run_test_on_X(X_train, clf)
    test_accuracy, test_loose_accuracy = run_test_on_X(X_test, clf)
    verify_accuracy, verify_loose_accuracy = run_test_on_X(X_verify, clf)
    print('[iter %d]train:%g, test: %g, verify: %g' % (i, accuracy, test_accuracy, verify_accuracy))
    print('[loose]  train:%g, test: %g, verify: %g' % ( loose_accuracy, test_loose_accuracy, verify_loose_accuracy))
    sum_accuracy += accuracy
    sum_test_accuracy += test_accuracy
    sum_verify_accuracy += verify_accuracy
    sum_loose_accuracy += loose_accuracy
    sum_loose_test_accuracy += test_loose_accuracy
    sum_loose_verify_accuracy += verify_loose_accuracy
    if accuracy + test_accuracy > best_accuracy:
        best_accuracy = accuracy + test_accuracy
        best_model = clf

print(sum_accuracy/n_run)
print(sum_test_accuracy/n_run)
print(sum_verify_accuracy/n_run)
print('loose')
print(sum_loose_accuracy/n_run)
print(sum_loose_test_accuracy/n_run)
print(sum_loose_verify_accuracy/n_run)

print('start')
print(best_model.startprob_)
print('trans')
print(best_model.transmat_)
print('emission')
print(best_model.emissionprob_)


def plot_on_bars(bars, name):
    close_seq = list(map(lambda x: x.close, bars))
    date_seq = list(map(lambda x: x.date, bars))
    train_diff = build_close_diff_seq(close_seq)
    train_diff_int = transform_to_int(train_diff, seg_edge)
    X_train, Y_train = transform_to_frames(train_diff_int, frame_length)

    predicts = []
    for test_seq in X_train:
        predicts.append(predict_next(test_seq, best_model))

    Date = pd.to_datetime(date_seq[frame_length + 1:]).date
    close = np.array(close_seq[frame_length + 1:])

    plt.figure(name, figsize=(25, 18))
    for i in range(n_value_seg):
        pos = (np.array(predicts) == i)
        plt.plot_date(Date[pos], close[pos], 'o', label='state %d' % i, linewidth=2)
        plt.legend()
    plt.draw()


plot_on_bars(train_bars, 'train')

plot_on_bars(verify_bars, 'verify')

plt.show()