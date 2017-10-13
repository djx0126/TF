import math
import numpy as np

from HMM.Bar import Bar
from hmmlearn.hmm import MultinomialHMM

train_file_name = 'rb88_daily.txt'
verify_file_name = 'rb1801_daily.txt'


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
verify_close_seq = list(map(lambda x: x.close, verify_bars))


def build_close_diff_seq(seq):
    diff_seq = []
    for i in range(len(seq)):
        if i > 0:
            diff_value = seq[i] - seq[i - 1]
            diff_value_ratio = diff_value / seq[i - 1]
            log_diff_value_ratio = math.log(diff_value_ratio + 1)
            diff_seq.append(log_diff_value_ratio)
    return diff_seq


train_diff = build_close_diff_seq(train_close_seq)
verify_diff = build_close_diff_seq(verify_close_seq)

n_value_seg = 5


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

ratio_for_test = 0.75
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
frame_length = 16


def transform_to_frames(value_array, frame_length):
    length = len(value_array)
    start = frame_length - 1
    X_frame = []
    Y_frame = []
    for i in range(start, length - frame_length):  # reserve one for Y
        X_frame.append(value_array[i:i + frame_length + 1])
        Y_frame.append(value_array[i + frame_length:i + frame_length + 1])
    return X_frame, Y_frame


X_train, Y_train = transform_to_frames(train_diff_int, frame_length)
X_test, Y_test = transform_to_frames(test_diff_int, frame_length)
X_verify, Y_verify = transform_to_frames(verify_diff_int, frame_length)
print(X_train[-3:])
print(Y_train[-3:])


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

n_components = 5


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
    clf = MultinomialHMM(n_components=n_components, n_iter=500)
    clf.fit(X_train_hmm, lengths=X_train_lengths)
    return clf;

def run_test_on_X(X, clf):
    right = 0
    for test_seq in X:
        expected = test_seq[len(test_seq) - 1]
        predict = predict_next(test_seq, clf)
        if predict == expected:
            right += 1

    accuracy = right / len(X)
    return accuracy


sum_accuracy = 0
sum_test_accuracy = 0
sum_verify_accuracy = 0
n_run = 10
for i in range(n_run):
    clf = train_on_X(X_train)
    accuracy = run_test_on_X(X_train, clf)
    test_accuracy = run_test_on_X(X_test, clf)
    verify_accuracy = run_test_on_X(X_verify, clf)
    print('train:%g, test: %g, verify: %g' % ( accuracy, test_accuracy, verify_accuracy))
    sum_accuracy += accuracy
    sum_test_accuracy += test_accuracy
    sum_verify_accuracy += verify_accuracy

print(sum_accuracy/n_run)
print(sum_test_accuracy/n_run)
print(sum_verify_accuracy/n_run)

