from utils.parse import *
from utils.gowalla import *

gowalla_path = "../Recommend/data/gowalla/Gowalla_totalCheckins.txt"


def TestFourSquare():
    dataset = FourSquareDataSet(
        "./data/square/checkin_CA_venues.txt", 0.2, False)
    # dataset.get_poi_2_poi_data(True,30)
    # dataset.get_poi_2_time_data(True)
    dataset.parse()


def TestGowalla():
    gowalla = GowallaParser(gowalla_path, 0.2)
    gowalla.parse()


if __name__ == "__main__":
    TestFourSquare()
    # TestGowalla()
