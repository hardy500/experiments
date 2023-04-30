#%%
import requests, zipfile, io
import unicodedata
import string
import seaborn  as sns
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional  as F
from torch.utils.data import Dataset, DataLoader
#%%

zip_file_url = "https://download.pytorch.org/tutorial/data.zip"
r = requests.get(zip_file_url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()
# %%

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
vocab = {v:k for k, v in enumerate(all_letters)}

# %%

def pad_and_pack(batch: List[Tuple]):
  input_tensors, labels, lengths = [], [], []
  for x, y in batch:
    input_tensors.append(x)
    labels.append(y)
    lengths.append(x.shape[0])

  x_padded = nn.utils.rnn.pad_sequence(input_tensors, batch_first=False)
  x_packed = nn.utils.rnn.pack_padded_sequence(x_padded, lengths, batch_first=False, enforce_sorted=False)
  y_batched = torch.as_tensor(labels, dtype=torch.long)
  return x_packed, y_batched

def unicode_to_ascii(s):
  return ''.join(
    c for c in unicodedata.normalize('NFD', s)
    if unicodedata.category(c) != 'Mn'
    and c in all_letters
  )

# %%
name_language_data = {}

for zip_path in z.namelist():
  if 'data/names' in zip_path and zip_path.endswith('.txt'):
    lang = zip_path[len('data/names'):-len('.txt')]
    with z.open(zip_path) as f:
      lang_names = [unicode_to_ascii(line).lower() for line in str(f.read(), encoding='utf-8').strip().split('\n')]
      name_language_data[lang] = lang_names
# %%

class EmbeddingPackable(nn.Module):
  """
  The embedding layer in pytorch does not support packed seq obj.
  This wrapper class will fix that
  """
  def __init__(self, embd_layer):
    super().__init__()
    self.embd_layer = embd_layer

  def forward(self, x):
    if isinstance(x, nn.utils.rnn.PackedSequence):
      sequences, lengths = nn.utils.rnn.pad_packed_sequence(x.cpu(), batch_first=True)
      sequences = self.embd_layer(sequences.to(x.data.device))
      return nn.utils.rnn.pack_padded_sequence(sequences, lengths.cpu(), batch_first=True, enforce_sorted=False)
    else:
      return self.embd_layer(x)

class LanguageNameDataset(Dataset):
  def __init__(self, lang_name_dict, vocabulary):
    self.label_names = [x for x in lang_name_dict.keys()]
    self.data = []
    self.labels = []
    self.vocabulary = vocabulary
    for y, language in enumerate(self.label_names):
      for sample in lang_name_dict[language]:
        self.data.append(sample)
        self.labels.append(y)

  def __len__(self):
    return len(self.data)

  def string2InputVec(self, input_string):
    """
    This method will convert any input string into a vector of long values, according to the vocabulary used by this object.
    input_string: the string to convert to a tensor
    """
    T = len(input_string) #How many characters long is the string?

    #Create a new tensor to store the result in
    name_vec = torch.zeros((T), dtype=torch.long)
    #iterate through the string and place the appropriate values into the tensor
    for pos, character in enumerate(input_string):
        name_vec[pos] = self.vocabulary[character]

    return name_vec

  def __getitem__(self, idx):
    name = self.data[idx]
    label = self.labels[idx]

    #Conver the correct class label into a tensor for PyTorch
    label_vec = torch.tensor([label], dtype=torch.long)

    return self.string2InputVec(name), label
# %%

dataset = LanguageNameDataset(name_language_data, vocab)
train_data, test_data = torch.utils.data.random_split(dataset, (len(dataset)-300, 300))
BS = 32
train_loader = DataLoader(train_data, batch_size=BS, shuffle=True, collate_fn=pad_and_pack)
test_loader = DataLoader(test_data, batch_size=BS, shuffle=False, collate_fn=pad_and_pack)
# %%

class LastTimeStep(nn.Module):
  def __init__(self, rnn_layers=1, bidirectional=False):
    super().__init__()

    self.rnn_layers = rnn_layers
    if bidirectional:
      self.num_directions = 2
    else:
      self.num_directions = 1

  def forward(self, input):
    # for input (out, h_t)
    rnn_out = input[0]
    last_step = input[1]
    # for input (out, (h_t, c_t))
    if isinstance(last_step, tuple):
      last_step = last_step[0]

    batch_size = last_step.shape[1]
    last_step = last_step.view(self.rnn_layers, self.num_directions, batch_size, -1)
    last_step = last_step[self.rnn_layers-1]
    last_step = last_step.permute(1, 0, 2)
    return last_step.reshape(batch_size, -1)
# %%

D = 64 # size of embedding result
vocab_size = len(all_letters)
hidden_node = 256 # number of neurons
classes = len(dataset.label_names)

first_rnn = nn.Sequential(
  EmbeddingPackable(nn.Embedding(vocab_size, D)), # (B, T) -> (B, T, D)
  nn.RNN(D, hidden_node, batch_first=True, bidirectional=True), # (B, T, D) -> ((B, T, D), (S, B, D))
  LastTimeStep(bidirectional=True),
  nn.Linear(hidden_node*2, classes) # (B, D) -> (B, classes); hidden_node douple in size when using bidirectional
)
# %%

from idlmam import train_simple_network
from sklearn.metrics import accuracy_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

loss_func = nn.CrossEntropyLoss()
batch_one_train = train_simple_network(
  first_rnn,
  loss_func,
  train_loader,
  test_loader,
  score_funcs={'Accuracy': accuracy_score},
  device=device,
  epochs=20
)

sns.lineplot(x='epoch', y='test Accuracy', data=batch_one_train, label='RNN');

# %%
