import pandas as pd
import os
import random
import mmengine


def random_slt_eq_subset(file: str, num: int) -> None:

    badminton_data_dict = {}
    slt_data = []

    if not os.path.exists(file):
        print(f"{file} does not exist.")
    
    else:
        if file.lower().endswith(('.pkl')):
            save_path = f"{file.split('.')[0]}_{num}_per_class.pkl"
            unpickled_df = pd.read_pickle(file)

            for sample in unpickled_df:
                if sample["label"] not in badminton_data_dict:
                    badminton_data_dict[sample["label"]] = [sample]
                else:
                    badminton_data_dict[sample["label"]].append(sample)

            for idx, item in badminton_data_dict.items():
                print(f"class {idx}: {len(item)}")
                for elm in random.sample(item, num):
                    slt_data.append(elm)

            mmengine.dump(slt_data, save_path)

            print(f'{save_path} is saved. ')

        elif file.lower().endswith(('.txt')):

            save_path = f"{file.split('.')[0]}_{num}_per_class.txt"
            with open(file) as f:
                lines = f.readlines()

            for l in lines:
                if l.split(' ')[1].rstrip() not in badminton_data_dict:
                    badminton_data_dict[l.split(' ')[1].rstrip()] = [l]
                else:
                    badminton_data_dict[l.split(' ')[1].rstrip()].append(l)
            
            for idx, item in badminton_data_dict.items():
                print(f"class {idx}: {len(item)}")
                for elm in random.sample(item, num):
                    slt_data.append(elm)

            with open(save_path, 'w') as file:
                for item in slt_data:
                    file.write(item)
            
            print(f'{save_path} is saved. ')

        else:
            print(f"The type of {file} is not valid.")

        


file = "/home/jovyan/local/mmaction2_badminton/badminton_dataset_ncu_coach_train_labels.txt"
random_slt_eq_subset(file, 50)

