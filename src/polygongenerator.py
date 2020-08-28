from shapely.geometry import Polygon
import numpy as np


def checkoverlap(rect1, rect2) :
    try :
        p1 = Polygon(
            [
                (rect1[0], rect1[1]),
                (rect1[1], rect1[1]),
                (rect1[2], rect1[3]),
                (rect1[2], rect1[1]),
            ]
        )
        p2 = Polygon(
            [
                (rect2[0], rect2[1]),
                (rect2[1], rect2[1]),
                (rect2[2], rect2[3]),
                (rect2[2], rect2[1]),
            ]
        )
        return p1.intersects(p2)

    except :
        return True

def getpolygons():
    x11 = np.random.uniform()
    y11 = np.random.uniform()
    x21 = np.random.uniform()
    y21 = np.random.uniform()

    x12 = np.random.uniform()
    y12 = np.random.uniform()
    x22 = np.random.uniform()
    y22 = np.random.uniform()

    rect1 = np.array([x11, y11,
                      x21, y21])

    rect2 = np.array([x12, y12,
                      x22, y22])

    if not checkoverlap(rect1, rect2) :
        return rect1, rect2


if __name__=="__main__":
    stack = []
    for i in range(200):
        get