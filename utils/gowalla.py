import pandas as pd
import math
import time
import pickle
from utils.common import *


class GowallaParser(object):
    def __init__(self, path, frac):
        self.df = pd.read_csv("./data/gowalla/Gowalla_totalCheckins.txt", sep="\t", header=None, names=[
                              "uid", "date", "latitude", "longitude", "location_id"])
        self.frac = frac
        self.path = "./data/gowalla/"
        self.decay = 0.19898883519701627

    def split(self, all_list, ratio=0.2):
        num = len(all_list)
        offset = int(num * ratio)
        if num == 0 or offset < 1:
            return [], all_list
        train = all_list[offset:]
        test = all_list[:offset]
        return train, test

    def parse(self):
        data = self.df
        user_item_map = {}
        uid_idx = 0
        uid_map = {}
        idIdx = 0
        id_map = {}
        venue_id_map = {}
        # 根据id group，同时替换POI id
        print("group-----")
        for idx, row in data.iterrows():
            userID = row["uid"]
            if userID == 2500:
                break
            if userID not in user_item_map:
                user_item_map[userID] = []
            user_item_map[userID].append([row['date'], row["location_id"]])
        # 去除 数据集合小于 10 的 数据行，同时对 uid进行重新map
        filter_user_map = {}

        for uid, l in user_item_map.items():
            if len(l) < 10:
                continue
            r = []
            for item in l:
                if item[1] not in id_map:
                    id_map[item[1]] = idIdx
                    venue_id_map[idIdx] = [item[1]]
                    idIdx += 1
                r.append([item[0], id_map[item[1]]])
            filter_user_map[uid_idx] = r
            uid_map[uid_idx] = [uid]
            uid_idx += 1

        # 分开数据
        print("split-----")
        train_map, test_map = {}, {}
        train_map_item = {}
        for k, l in filter_user_map.items():
            train, test = self.split(l)
            train_map_item[k] = train
            train_map[k] = [l[1] for l in train]
            test_map[k] = [l[1] for l in test]
        self.write_map(train_map, "train")
        self.write_map(test_map, "test")
        self.write_map(uid_map, "uid")
        self.write_map(venue_id_map, "gowalla")

        # 权重表
        print("count score ---")
        train_score = {}
        for uid, l in train_map_item.items():
            # for debug
            if uid == 0:
                for item in l:
                    print(item[0])
            # 基准时间
            time_stamp = time.mktime(
                time.strptime(l[0][0], '%Y-%m-%dT%H:%M:%SZ'))
            if uid not in train_score:
                train_score[uid] = {}

            if l[0][1] not in train_score[uid]:
                train_score[uid][l[0][1]] = 1
            for his in l[1:]:
                t = time.mktime(time.strptime(his[0], '%Y-%m-%dT%H:%M:%SZ'))
                day = (time_stamp - t) / 86400.0
                # 衰减公式
                factor = count_factor(day)
                if his[1] not in train_score[uid]:
                    train_score[uid][his[1]] = factor
                else:
                    train_score[uid][his[1]] += factor
        with open(self.path + "scores.pkl", "wb") as f:
            pickle.dump(train_score, f, pickle.HIGHEST_PROTOCOL)
        return train_score

    def write_map(self, user_item_map, label):
        file = open(self.path + label + ".txt", 'w')
        for uid in sorted(user_item_map):
            items = user_item_map[uid]
            str_items = [str(uid)] + [str(i) for i in items]
            file.write(" ".join(str_items) + "\n")
        file.close()
