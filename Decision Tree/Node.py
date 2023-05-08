
class Node():
    def __init__(
        self,
        feature=None,
        threshold=None,
        value=None,
        left=None,
        right=None,
        info_gain=None
    ):

        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right
        self.info_gain = info_gain
