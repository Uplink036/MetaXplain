
def accuracy(targets, predictions):
    assert len(targets) == len(predictions)
    correct = 0
    for i in range(0, len(targets)):
        if targets[i] == predictions[i]:
            correct += 1
    return correct / len(targets)

def recall(targets, predictions):
    tp = {}
    fn = {}
    for i in range(0, 10):
        tp[i] = 0
        fn[i] = 0

    for i in range(0, len(targets)):
        if targets[i] == predictions[i]:
            tp[targets[i]] += 1
        else:
            fn[targets[i]] += 1

    recall = []
    for i in range(0, 10):
        recall.append(tp[i] / (tp[i] + fn[i]))
    return recall   

def precision(targets, predictions):
    tp = {}
    fp = {}
    for i in range(0, 10):
        fp[i] = 0
        tp[i] = 0

    for i in range(0, len(targets)):
        if targets[i] == predictions[i]:
            tp[targets[i]] += 1
        else:
            fp[predictions[i]] += 1

    precision = []
    for i in range(0, 10):
        precision.append(tp[i] / (tp[i] + fp[i]))
    return precision   
