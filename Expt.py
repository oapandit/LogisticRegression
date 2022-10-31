import numpy as np

from Models import LogisticRegression


def read_csv(csv_file_path):
    data = np.loadtxt(csv_file_path, delimiter=',', dtype=str)
    x, y = data[1:, 0:-1], data[1:, -1]
    return x, y


def main():
    csv_path = "C:/Users/opandit/Downloads/Data/data.csv"
    x, y = read_csv(csv_path)
    lr = LogisticRegression(learning_rate=0.01, reg='l1',is_verbose=True)
    lr.fit(x, y)
    print(lr.acc_score(x,y))


if __name__ == '__main__':
    main()
