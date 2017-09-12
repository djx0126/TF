import numpy as np

class InputData(object):
    def __init__(self, src_data, label_size, batch_size=0, name=None):
        self.__name = name
        self.__raw_data = np.array(src_data)
        self.M = len(src_data)
        self.__next_batch_index = 0
        self.__seq = np.arange(self.M)
        self.__batch_size = batch_size
        self.__label_size = label_size
        self.shuffle()

    def data(self):
        seq = self.__seq[0:self.M]
        return self.__raw_data[seq]

    @property
    def label_size(self):
        return self.__label_size

    def total_size(self):
        return self.M

    def shuffle(self):
        np.random.shuffle(self.__seq)
        self.__next_batch_index = 0

    def next(self, batch_size=None):
        if batch_size is None:
            batch_size = self.__batch_size
        if batch_size <= 0:
            return self.data()

        batch_start = self.__next_batch_index
        batch_end = batch_start + batch_size
        if (batch_end > self.M):
            self.shuffle()
            return self.next(batch_size)
        seq = self.__seq[batch_start:batch_end]
        batch = self.__raw_data[seq]
        self.__next_batch_index += batch_size
        return batch

    def random_pick(self, partial):
        if (partial > 0 and partial < 1):
            partial = int(partial * self.M)
            seq1 = self.__seq[0: partial]
            part1_data = self.__raw_data[seq1]
            part1 = InputData(src_data=part1_data, label_size=self.__label_size, batch_size= self.__batch_size)
            seq2 = self.__seq[partial: self.M]
            part2_data = self.__raw_data[seq2]
            part2 = InputData(src_data=part2_data, label_size=self.__label_size, batch_size= self.__batch_size)
            return (part1, part2)
        return self

if __name__ == '__main__':
    input_data = InputData(src_data=[[1,1], [2,2], [3,3], [4,4], [5,5], [6,6], [7,7], [8.8], [9.9], [10,10]], label_size=0)
    print('totol size: ' + str(input_data.total_size()))

    train, test = input_data.random_pick(0.8)
    print('train size: ' + str(train.total_size()) + ' items:' + str(train.data()))
    print('test size: ' + str(test.total_size()) + ' items:' + str(test.data()))

    data = input_data.data()
    print('data: ' + str(data.shape))

    def printNext(input_data, count=None):
        batch_data = input_data.next(count)
        next_item = 'print next ' + ' items:'
        for idx in range(len(batch_data)):
            next_item = next_item + ' ' + str(batch_data[idx])
        print(next_item)

    printNext(train, 3)
    printNext(train, 3)
    printNext(train, 3)
    printNext(train, 3)

    print('print next batch with data size = 5')
    train = InputData(src_data=train.data(), label_size=0, batch_size=5)
    printNext(train)

    print('print next batch with total data')
    test = InputData(src_data=test.data(), label_size=0)
    printNext(test)