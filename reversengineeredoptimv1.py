import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Define vocabulary and dataset
vocab = ["what", "is", "the", "of", "2", "3", "4", "5", "7", "8", "10", 
         "capital", "color", "sky", "opposite", "up", "down", "france", "paris", "blue"]

vocab_size = len(vocab)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

dataset = [
    ("what is 2+2", "4"),
    ("what is the capital of france", "paris"),
    ("what color is the sky", "blue"),
    ("what is 3+5", "8"),
    ("what is 10-7", "3"),
    ("what is the opposite of up", "down"),
]

class QA_Dataset(Dataset):
    def __init__(self, data, word_to_idx):
        self.data = data
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, answer = self.data[idx]
        q_tokens = [self.word_to_idx[word] for word in question.split()]
        a_token = self.word_to_idx[answer]
        return torch.tensor(q_tokens), torch.tensor(a_token)

train_dataset = QA_Dataset(dataset, word_to_idx)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Define transformer model
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        embedded = embedded.permute(1, 0, 2)  # (seq_len, batch, embed_dim)
        transformer_output = self.transformer(embedded)  # (seq_len, batch, embed_dim)
        output = self.fc(transformer_output[-1])  # Use the final token output
        return output

# Hyperparameters
embed_dim = 16
num_heads = 2
num_layers = 2
hidden_dim = 32
model = SimpleTransformer(vocab_size, embed_dim, num_heads, num_layers, hidden_dim)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # Adjust epochs for experimentation
    total_loss = 0
    for q_tokens, a_token in train_loader:
        optimizer.zero_grad()
        output = model(q_tokens)
        loss = criterion(output, a_token)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Test the model
def predict(question):
    q_tokens = [word_to_idx[word] for word in question.split()]
    q_tensor = torch.tensor(q_tokens).unsqueeze(0)
    with torch.no_grad():
        output = model(q_tensor)
        predicted_idx = output.argmax(dim=1).item()
        return vocab[predicted_idx]

test_question = "what is 2+2"
print(f"Question: {test_question}, Predicted Answer: {predict(test_question)}")
