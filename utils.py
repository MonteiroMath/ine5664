    def accuracy(predictions, labels):
        return int(sum(labels == predictions) / len(labels) * 100)
