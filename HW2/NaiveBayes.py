import csv
import random
import math
import pickle
import numpy as np


def get_data():
    train = []
    test = []
    inp = list(csv.reader(open('spambase.data', 'r')))

    # convert all to floats
    for i in range(len(inp)):
        for j in range(len(inp[i])):
            inp[i][j] = float(inp[i][j])

    # shuffle to maintain randomness
    random.shuffle(inp)

    # get the max from each field
    ma = []
    for i in zip(*inp):
        ma.append(max(i))

    # normalize data and offset the median to 1 to 2
    # also divide into train and test data
    for i in range(len(inp)):
        r = inp[i]
        for j in range(len(r) - 1):
            r[j] = r[j] / ma[j] + 1.0
        if i < 2000:
            train.append(r)
        else:
            test.append(r)
    return train, test


class NaiveBayes:
    values = {}
    quanta = {}
    accuracy = 0

    # the specific dataset has y as the last value
    # split the data for finding the probability for each field
    def populate(self, data):
        self.values.clear()
        self.quanta.clear()
        self.accuracy = 0

        # this uniques the list of classes
        classes = list(set(list(zip(*data))[-1]))

        # populate the training data into the class
        for i in classes:
            self.values[i] = []
        for i in range(len(data)):
            self.values[data[i][-1]].append(data[i])
        return self

    # populate the probabilities for the dataset
    def train(self):
        # get the means and standard deviations for each of the classes in values
        self.quanta.clear()
        # class level
        vals = list(self.values.values())
        inv_vals = []
        for i in vals:
            inv_vals.append(list(zip(*i)))
        for i, k in zip(inv_vals, range(len(inv_vals))):
            # field level: gather the probability of a value being at a threshold
            fields = []
            for j in i[:-1]:
                mean = np.mean(j)
                stddev = np.std(j)
                fields.append({'mean': mean, 'stddev': stddev})

            # store the quantization of the fields for the classification
            self.quanta[k] = fields
        return self

    def evaluate(self, data):
        res = []
        # get the probability of test for each field
        correct = 0
        for item in data:
            prob_max = 0
            class_max = 0
            for classification in self.quanta:
                prob = 1
                for field, j, index in zip(item[:-1], self.quanta[classification], range(len(item[:-1]))):  # the -1 removes the label
                    prob *= self.get_gaussian(field, j['mean'], j['stddev'])
                if prob > prob_max:
                    prob_max = prob
                    class_max = classification
            if class_max == item[-1]:
                correct += 1
        self.accuracy = correct / float(len(data)) * 100
        return self

    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    def load(self, filename):
        self.__dict__.update(pickle.load(open(filename, 'rb')))

    # Gaussian: pdf of norm distribution of random vars
    # gaussian e^((x-mean)^2/(2*stdev^2)) / sqrt(2 pi) * stdev
    @staticmethod
    def get_gaussian(x, mean, stdev):
        e_top = math.pow(x-mean,2)
        e_bottom = 2*math.pow(stdev,2)
        if e_bottom <= 0.0:
            return 0
        e = math.exp(-(e_top/e_bottom))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * e


n = NaiveBayes()
train, test = get_data()
n.populate(train)
n.train()
n.evaluate(test)
print(n.accuracy)

# Note that step 4.A is done in the get_data method on line 31
