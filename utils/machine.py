import math

class Machine:
    def __init__(self, max_resource):
        self.max_resource = max_resource

    def machine1(self, x):
        max_resource = self.max_resource
        point1 = max_resource / 3
        point2 = max_resource / 3 * 2
        if x < point1:
            return math.sin(x * (3 / 2 * math.pi / max_resource)) * point1
        elif x < point2:
            return x
        elif x < max_resource:
            return 2 * (x - point1)
        else:
            return 2 * (max_resource - point1)

    def machine2(self, x):
        max_resource = self.max_resource
        point1 = max_resource / 3 * 2
        if x < point1:
            return point1 / (
                1 + math.exp(-10 * (x - (max_resource / 3)) / max_resource)
            )
        elif x < max_resource:
            return max_resource * (1 + math.cos(math.pi / point1 * x)) + point1 / (
                1 + math.exp(-(point1 - (max_resource / 3)) / max_resource * 10)
            )
        else:
            return max_resource * (
                1 + math.cos(math.pi / point1 * max_resource)
            ) + point1 / (
                1 + math.exp(-(point1 - (max_resource / 3)) / max_resource * 10)
            )

    def machine3(self, x):
        max_resource = self.max_resource
        point1 = max_resource / 3
        point2 = max_resource / 3 * 2
        if x < point1:
            return x
        elif x < point2:
            return 8 / 5 * (x - point1) + point1
        elif x < max_resource:
            return (x - point2) * 5 / 6 + (8 / 5 * (point2 - point1) + point1)
        else:
            return (max_resource - point2) * 5 / 6 + (
                8 / 5 * (point2 - point1) + point1
            )

    def machine4(self, x):
        max_resource = self.max_resource
        point1 = max_resource / 4
        point2 = max_resource / 2
        point3 = max_resource / 4 * 3
        if x < point1:
            return 0
        elif x < point2:
            return 3 * (x - point1)
        elif x < point3:
            return math.sin(
                (x - point2) * (2 * math.pi / max_resource)
            ) * point1 + 3 * (point2 - point1)
        elif x < max_resource:
            return (
                8 / 5 * (x - point3)
                + math.sin((point3 - point2) * (2 * math.pi / max_resource)) * point1
                + 3 * (point2 - point1)
            )
        else:
            return (
                8 / 5 * (max_resource - point3)
                + math.sin((point3 - point2) * (2 * math.pi / max_resource)) * point1
                + 3 * (point2 - point1)
            )

    def get_funclist(self, numlist):
        funcs = [
            self.machine1,
            self.machine1,
            self.machine2,
            self.machine3,
            self.machine4,
        ]
        return [funcs[i] for i in numlist]

