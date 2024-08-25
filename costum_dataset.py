from torch.utils.data import Dataset, DataLoader
import re
import random

seed_value = 42  # Vous pouvez choisir n'importe quel nombre entier comme seed

random.seed(seed_value)

class CustomDataset(Dataset):
    def __init__(self, file_path, ration_train):
        self.file_path = file_path
        samples = []
        
        with open(file_path, 'r') as f:
            for line in f:
                samples.append(line.strip())

        random.shuffle(samples)

        split_index = int(len(samples) * ration_train)
        # Diviser la liste en deux sous-listes
        self.train_data = samples[:split_index]
        self.test_data = samples[split_index:]

        self.train = True 
        random.seed()


    def switch_test(self,bool) : 
        self.train = not bool

    def __len__(self):

        if self.train : 
            samples = self.train_data
        else : 
            samples = self.test_data

        return len(samples)

    def __getitem__(self, idx):

        if self.train : 
            samples = self.train_data
        else : 
            samples = self.test_data

        id_part, query_part = self.process_query(samples[idx])
        return {
            'query_id': id_part,
            'query': query_part
        }
    
    def process_query(self, query):
        match = re.match(r"(2024-\d+)\t(.+)", query)

        part1 = match.group(1) 
        part2 = match.group(2)

        return part1, part2
    
def collate_fn(batch):
    collated_batch = []
    for item in batch:
        collated_batch.append({'query_id': item['query_id'], 'query': item['query']})
    return collated_batch
