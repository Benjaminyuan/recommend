import pickle
import numpy as np
import pandas as pd
import calendar
import time
import random
import math
from utils.common import *

month_dict = dict((v, k) for k, v in enumerate(calendar.month_abbr))


def get_strline_out(str_list):
    return ' '.join([str(x) for x in str_list]) + '\n'


def get_words(words_str):
    words = words_str[1:-1].split(',')[:-1]
    return [x.replace(' ', '') for x in words]


def write_in_file(f_out_path, str_out):
    f_out = open(f_out_path, 'w')
    f_out.write(str_out)
    f_out.close()


class GMTTimer(object):

    def __init__(self, time_str):
        self.week = time_str[0:3]
        self.month = time_str[4:7]
        self.day = time_str[8:10]
        self.hour = time_str[11:13]
        self.minute = time_str[14:16]
        self.second = time_str[17:19]
        self.year = time_str[-4:]

    def get_time_slot(self):
        return (self.hour + '1') if self.week in ["Sat", "Sun"] else (self.hour + '0')

    def get_timestamp(self):
        format_time = self.year + '-' + \
            str(month_dict[self.month]) + '-' + self.day + ' ' + \
            self.hour + ':' + self.minute + ':' + self.second
        return time.mktime(time.strptime(format_time, "%Y-%m-%d %H:%M:%S"))


class FourSquareDataSet(object):
    def __init__(self, path, frac, debug=False):
        self.id_map = {}
        self.venue_id_map = {}
        self.all_data = {}
        self.id_idx = 0
        self.frac = frac
        # self.decay = 0.19898883519701627
        df = pd.read_csv(path, header=0, sep='\t')
        self.path = "./data/square/"
        self.df = df
        # self.train_data = file_data.sample(frac=frac, random_state=0, axis=0)
        # self.test_data = file_data[~file_data.index.isin(
        #     self.train_data.index)]
        # # self.poi_set = file_data.drop_duplicates(subset=["VenueId"])["VenueId"]
        # self.poi_set = file_data["VenueId"].drop_duplicates()
        # self.n_items = len(self.poi_set)
        # if debug:
        #     self.train_data = self.train_data.head(500)
        #     self.test_data = self.test_data.head(500)
        # self.n_users = self.train_data["userID"].drop_duplicates()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_poi_2_time_data(self, write):
        poi_2_time_map = {}
        for idx, row in self.train_data.iterrows():
            poiId = row["VenueId"]
            time_slot = GMTTimer(row["Time(GMT)"]).get_time_slot()
            if poiId not in poi_2_time_map:
                poi_2_time_map[poiId] = {time_slot: 1}
            elif time_slot in poi_2_time_map[poiId]:
                poi_2_time_map[poiId][time_slot] += 1
            else:
                poi_2_time_map[poiId][time_slot] = 1
        self.poi_time_list = []
        for venueId, time_slot_map in poi_2_time_map.items():
            for time_slot, weight in time_slot_map.items():
                self.poi_time_list.append([venueId, time_slot, weight])
        # 写入csv文件
        if write:
            df = pd.DataFrame(self.poi_time_list, columns=[
                              "venue_id", "time_slot", "weight"])
            df.to_csv("poi_time.csv", index=False)

    def get_poi_2_poi_data(self, write, delta):
        uid_set = self.train_data["userID"].drop_duplicates()
        poi_2_poi_map = {}
        interval = delta * 24 * 3600
        for uid in uid_set:
            uid_data = self.train_data[self.train_data['userID'] == uid]
            uid_data_list = []
            # 获取用户的行为列表
            for idx, row in uid_data.iterrows():
                uid_data_list.append(
                    [row["VenueId"], GMTTimer(row["Time(GMT)"]).get_timestamp()])
                print(uid_data_list[len(uid_data_list) - 1])
            # 对行为列表的每一个进行遍历
            for idx, [venueId, timestamp] in enumerate(uid_data_list):
                for idx1, [venueId1, timestamp1] in enumerate(uid_data_list[idx+1:]):
                    if timestamp1 < timestamp and timestamp - timestamp1 < interval:
                        # 按字典序组装key
                        key = (
                            venueId + "," + venueId1) if venueId > venueId1 else (venueId1 + "," + venueId)
                        if key in poi_2_poi_map:
                            poi_2_poi_map[key] += 1
                        else:
                            poi_2_poi_map[key] = 1
        self.poi_2_poi_list = []
        for key, weight in poi_2_poi_map.items():
            venueIds = key.split(",")
            self.poi_2_poi_list.append([venueIds[0], venueIds[1], weight])
        if write:
            df = pd.DataFrame(self.poi_2_poi_list, columns=[
                              "venue_id1", "venue_id2", "weight"])
            df.to_csv("poi_poi.csv", index=False)

    def get_user_item_data(self):
        self.group_raw_data(self.train_data, "train")
        self.group_raw_data(self.test_data, "test")
        self.save_venue()

    def group_raw_data(self, data, label):
        user_item_map = {}
        idIdx = self.id_idx
        id_map = self.id_map
        venue_id_map = self.venue_id_map
        for idx, row in data.iterrows():
            userID = row["userID"]
            poiId = row["VenueId"]
            if userID not in user_item_map:
                user_item_map[userID] = []
            if poiId not in id_map:
                id_map[poiId] = idIdx
                venue_id_map[idIdx] = poiId
                idIdx += 1
            user_item_map[userID].append(id_map[poiId])

    def write_map(self, user_item_map, label):
        file = open("./data/square/" + label + ".txt", 'w')
        for uid in sorted(user_item_map):
            items = user_item_map[uid]
            str_items = [str(uid)] + [str(i) for i in items]
            file.write(" ".join(str_items) + "\n")
        file.close()

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
            userID = row["userID"]
            if userID not in user_item_map:
                user_item_map[userID] = []
            user_item_map[userID].append(
                [row['Time(GMT)'], row["VenueId"], row["VenueCategory"]])
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
                r.append([item[0], id_map[item[1]], item[2]])
            filter_user_map[uid_idx] = r
            uid_map[uid_idx] = [uid]
            uid_idx += 1

        # 分开数据
        print("split-----")
        train_map, test_map = {}, {}
        train_map_item = {}
        poi_c = {}
        for k, l in filter_user_map.items():
            train, test = self.split(l)
            train_map_item[k] = train
            train_map[k] = [l[1] for l in train]
            test_map[k] = [l[1] for l in test]
            # poi - c
            for item in train:
                poi_c[item[1]] = item[2]
        self.write_map(train_map, "train")
        self.write_map(test_map, "test")
        self.write_map(uid_map, "uid")
        self.write_map(venue_id_map, "square")
        # 权重表
        print("count score ---")
        train_score = {}
        for uid, l in train_map_item.items():
            # 基准时间
            time_stamp = GMTTimer(l[0][0]).get_timestamp()
            if uid not in train_score:
                train_score[uid] = {}

            if l[0][1] not in train_score[uid]:
                train_score[uid][l[0][1]] = 1
            for his in l[1:]:

                t = GMTTimer(his[0]).get_timestamp()
                day = (time_stamp - t) / 86400.0
                if uid <= 5:
                    print("uid: %d poi: %d,cur: %s,day: %d," %
                          (uid, his[1], his[0], day))
                # 衰减公式
                # factor = math.exp(- max((day - 7), 0) / 7 * self.decay)
                factor = count_factor(day)
                if his[1] not in train_score[uid]:
                    train_score[uid][his[1]] = factor
                else:
                    train_score[uid][his[1]] += factor
        with open(self.path + "scores.pkl", "wb") as f:
            pickle.dump(train_score, f, pickle.HIGHEST_PROTOCOL)

        # 计算poi-category的映射
        print('poi_word_gen----')
        data = poi_c
        length = len(data)
        poi_word_tf_dict = {}
        word_numinpoi_dict = {}
        for poiID, cate in data.items():
            words = get_words(cate)
            words_length = len(words)
            for word in words:
                if poiID not in poi_word_tf_dict:
                    poi_word_tf_dict[poiID] = {word: 1/words_length}
                else:
                    if word not in poi_word_tf_dict[poiID]:
                        poi_word_tf_dict[poiID][word] = 1/words_length
                    else:
                        print('poi_word_gen错误(poiID,word)', poiID, word)
                if word not in word_numinpoi_dict:
                    word_numinpoi_dict[word] = 1
                else:
                    word_numinpoi_dict[word] += 1
        str_out = ''
        for poiID, value in poi_word_tf_dict.items():
            for word, tf in poi_word_tf_dict[poiID].items():
                str_out += get_strline_out([poiID, word, tf *
                                           math.log(length/(word_numinpoi_dict[word]+1))])
                str_out += get_strline_out([word, poiID, tf *
                                           math.log(length/(word_numinpoi_dict[word]+1))])
        write_in_file('./data/square/net_POI_word.txt', str_out)
        return train_score

    def split(self, all_list, ratio=0.2):
        num = len(all_list)
        offset = int(num * ratio)
        if num == 0 or offset < 1:
            return [], all_list
        train = all_list[offset:]
        test = all_list[:offset]
        return train, test
