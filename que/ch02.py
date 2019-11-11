path = '/Users/tnnque/PycharmProjects/data-analytics-ex/datasets/bitly_usagov/example.txt'
print(open(path).readline())

import json
path = '/Users/tnnque/PycharmProjects/data-analytics-ex/datasets/bitly_usagov/example.txt'
records = [json.loads(line) for line in open(path)]
# print(records[0]['tz'])
# Counting Time Zones
time_zones = [rec['tz'] for rec in records if 'tz' in rec]
print(time_zones[:10])

# def get_counts(sequence):
#     counts = dict()
#     for c in sequence:
#         if c not in counts:
#             counts[c] = 1
#         else:
#             counts[c] += 1
#     return counts
#
# from collections import defaultdict
# def get_counts2(sequence):
#     counts = defaultdict(int)
#     for c in sequence:
#         counts[c] += 1
#     return counts
#
# counts = get_counts2(time_zones)
# print(counts['America/New_York'])
# print(len(time_zones))
#
# def top10 (count_dict, n = 10):
#     pair = [(count, tz) for tz, count in count_dict.items()]
#     pair.sort()
#     return pair[-n:]
# print(top10(counts))

# from collections import Counter
# counts = Counter(time_zones)
# print(counts.most_common(10))

from pandas import DataFrame, Series
import pandas as pd
frame = DataFrame(records)
# print(frame)

# print(frame['tz'][:10])

tz_counts = frame['tz'].value_counts()
# print(tz_counts[:10])

clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_counts = clean_tz.value_counts()

tz_counts[:10].plot(kind= 'barh', rot = 0)
# print(frame['a'][51])

results = Series([x.split()[0] for x in frame.a.dropna()])
# print(results.value_counts()[:8])

import numpy as np
cframe = frame[frame.a.notnull()]
operating_system = np.where(cframe['a'].str.contains('Windows'), 'Windows', 'Not Windows')
print(operating_system[:5])

by_tz_os = cframe.groupby(['tz', operating_system])
agg_counts = by_tz_os.size().unstack().fillna(0)

indexer = agg_counts.sum(1).argsort()
print(indexer[:10])