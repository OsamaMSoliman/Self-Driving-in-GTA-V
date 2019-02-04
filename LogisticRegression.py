import matplotlib.pyplot as plt
import numpy as np


# continues range between 0 and 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# crossEntropy = summation(ln(probability of being what self actually is))
# x * ln(p) + (1-x) * ln(1-p) | x = 0 or 1
def crossEntropy(probability, label, numOfTotalPoints):
    return -(label @ np.log(probability) + (1-label) @ np.log(1 - probability)) / numOfTotalPoints


def gradientDescent(weights, points, label, learningRate, numOfIterations):
    numOfTotalPoints = points.shape[0]
    labelT = label.reshape(label.shape[0], 1)
    for i in reversed(range(numOfIterations)):
        probability = sigmoid(points @ weights)
        # calc the gradient and update the weights
        weights -= (learningRate / numOfTotalPoints) * (points.T @ (probability - labelT))

        w1, w2, b = weights
        pointsOnX = np.array([points[:, 0].min(), points[:, 0].max()])
        # w1X + w2Y + b => solve for Y
        pointsOnY = (-b - w1 * pointsOnX) / w2

        # print the error
        print(crossEntropy(probability, label, numOfTotalPoints))

        # line animation
        lines = plt.plot(pointsOnX, pointsOnY)
        if i != 0:
            plt.pause(0.0001 * i)
            lines[0].remove()


if __name__ == "__main__":
    # initialize the plot
    _, ax = plt.subplots(figsize=(4, 4))

    # training data size
    numOfPointsPerSet = 100
    # fixing the seed to reproduce the same results
    np.random.seed(0)

    # caching
    label0 = np.zeros(numOfPointsPerSet)
    label1 = np.ones(numOfPointsPerSet)
    set0CenterX, set0CenterY = 10, 12
    set0Deviation = 2
    set1CenterX, set1CenterY = 5, 6
    set1Deviation = 2

    # training data
    set0 = np.array([np.random.normal(set0CenterX, set0Deviation, numOfPointsPerSet),
                     np.random.normal(set0CenterY, set0Deviation, numOfPointsPerSet)]).T
    set1 = np.array([np.random.normal(set1CenterX, set1Deviation, numOfPointsPerSet),
                     np.random.normal(set1CenterY, set1Deviation, numOfPointsPerSet)]).T
    trainingData = np.vstack((set0, set1))
    bais = np.ones(trainingData.shape[0]).reshape(trainingData.shape[0], 1)
    trainingData = np.hstack((trainingData, bais))

    # plot the 2 sets
    ax.scatter(set0[:, 0], set0[:, 1], color='r')
    ax.scatter(set1[:, 0], set1[:, 1], color='k')

    gradientDescent(weights=np.ones(3).reshape(3, 1),
                    points=trainingData,
                    label=np.array([label0, label1]).reshape(numOfPointsPerSet * 2),
                    learningRate=0.01,
                    numOfIterations=1000)

    # show the plotted graph
    plt.show()
