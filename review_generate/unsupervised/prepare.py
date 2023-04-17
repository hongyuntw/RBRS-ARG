import glob
import os
import csv
import json
import random


mode = 'eval'
data_folder_path = f'review_generate/unsupervised/data/reviews/{mode}/'

"""
{
    'reviews' : list of reviews
    'target' : target of reviw (string)
}

"""
data = []

count = 0
target_star = 0

for file in glob.glob(data_folder_path + '*.csv'):
    print(file)
    fp = open(file)

    group_data = []
    rows = csv.reader(fp, delimiter='\t')
    for row in rows:
        group_data.append(row)

    """
    group_data[0] = 
    ['group_id', 'review_text', 'rating', 'category', 'rouge1', 'rouge2', 'rougeL', 'rating_dev']  
    """


    for _ in range(1):
        ri_idx = random.randint(1, 9)
        r_without_i = [group_data[i] for i in range(len(group_data)) if i != ri_idx and i != 0]
        ri = group_data[ri_idx]
        data.append({
            'reviews' : [x[1] for x in r_without_i],
            'reviews_rating' : [x[2] for x in r_without_i],
            'target' : ri[1],
            'target_rating' : ri[2]
        })
        count += 1
        target_star += float(ri[2])

with open(f'review_generate/unsupervised/data/{mode}.json', 'w', encoding='utf8') as f:
    json.dump(data, f, ensure_ascii=False)
