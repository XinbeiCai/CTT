import os
import pickle
import torch
import pandas as pd
from transformers import T5EncoderModel,AutoModel, AutoTokenizer
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-x', default='Books', help='category')
parser.add_argument('-y', default='Movies_and_TV', help='category')
parser.add_argument('-z', default='CDs_and_Vinyl', help='category')
args = parser.parse_args()



def review_emb(category_name):
    batch_size = 1024
    data_directory = f'../review_datasets/{category_name}/'
    meta_file_path = os.path.join(data_directory, f'{category_name}_meta.csv')

    # 加载模型
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 加载 T5 模型和 tokenizer（替换成你需要的模型，如 'bert-base-uncased'）
    model_name = "../multilingual-e5-large"  # 或 "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    # 读取数据
    df = pd.read_csv(meta_file_path, header=None, names=['item_id', 'information'])
    df.reset_index(inplace=True)

    # 分批处理
    information_embedding = []
    batch_num = (df.shape[0] + batch_size - 1) // batch_size  # 计算总批次数

    for i in tqdm(range(batch_num), total=int(batch_num), ncols=70, leave=False, unit='b'):
        if i == batch_num - 1:
            batch = df.loc[batch_size * i:, 'information'].tolist()
        else:
            batch = df.loc[batch_size * i:batch_size * (i + 1) - 1, 'information'].tolist()

        # 批量编码文本（核心改动）
        inputs = tokenizer(
            batch,  # 直接传入整个batch的文本列表
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)

        # e5 emb size 1024
        with torch.no_grad():
            #
            outputs = model(**inputs)
            # 平均池化（考虑attention mask）
            embeddings = outputs.last_hidden_state.mean(dim=1)
            information_embedding.extend(embeddings.tolist())

    # 保存结果
    df['meta_embedding'] = information_embedding
    df_information = df.drop(['information'], axis=1)
    assert len(information_embedding) == len(df), "嵌入数量与数据行数不匹配！"


    # InformationInfo = {}
    # for _, row in df_information.iterrows():
    #     itemid = int(row['item_id']) # 这里有问题
    #     information_embedding = row['meta_embedding']
    #     InformationInfo[itemid] = information_embedding
    #
    # # 写入 pickle 文件
    # with open(data_directory + f'{category_name}_meta_emb_t5.dat', 'wb') as f:
    #     pickle.dump(InformationInfo, f)
    InformationInfo = {}
    error_count = 0

    for _, row in df_information.iterrows():
        try:
            # 1. 强制转为字符串并去除首尾空格
            item_str = str(row['item_id']).strip()

            # 2. 检查是否为纯数字（允许负数可加 item_str.lstrip('-').isdigit()）
            if not item_str.isdigit():
                raise ValueError(f"非数字字符: {item_str}")

            # 3. 转换为整数
            itemid = int(item_str)
            InformationInfo[itemid] = row['meta_embedding']

        except Exception as e:
            error_count += 1
            print(f"警告：跳过非法item_id '{row['item_id']}' (错误: {str(e)})")
            continue

    if error_count > 0:
        print(f"总跳过记录数: {error_count}")

    # 写入文件（保持字典格式）
    with open(data_directory + f'{category_name}_meta_emb_e5.dat', 'wb') as f:
        pickle.dump(InformationInfo, f)

    print(f'meta_{category_name}_write_done!')


if __name__ == '__main__':
    review_emb(args.x)
    review_emb(args.y)
    review_emb(args.z)