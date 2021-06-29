from math import cos, asin, sqrt, pi


def distance(lat1, lon1, lat2, lon2):
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) *\
        cos(lat2*p) * (1-cos((lon2-lon1)*p))/2

    if a < 0:
        a = 0
    if sqrt(a) > 1 or sqrt(a) < 0:
        print(a)

    a = min(1, sqrt(a))
    a = max(0, a)
    return 12742 * asin(a)
