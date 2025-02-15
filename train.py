import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from model import FFNNLM, RNNLM, LSTMLM

class TextDataset(Dataset):
    def __init__(self, corpus_path, context_size=3):
        with open(corpus_path, 'r') as f:
            text = f.read()
        tokens = text.split()
        
        # Build vocabulary
        self.vocab = defaultdict(lambda: 0)  # Default to <unk>
        word_counts = defaultdict(int)
        for word in tokens:
            word_counts[word] += 1
        
        # Assign indices (0 preserved for <unk>)
        self.vocab['<unk>'] = 0
        for idx, (word, count) in enumerate(word_counts.items(), start=1):
            self.vocab[word] = idx
            
        # Convert tokens to indices
        self.data = []
        for i in range(len(tokens)-context_size+1):
            context = tokens[i:i+context_size-1]
            target = tokens[i+context_size-1]
            context_idx = [self.vocab[word] for word in context]
            target_idx = self.vocab.get(target, 0)
            self.data.append((torch.tensor(context_idx), torch.tensor(target_idx)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_model(corpus_path, model_type="f", context_size=3, embed_dim=100, hidden_dim=128, epochs=10, val_split=0.1):
    # Initialize dataset and split into train/val
    full_dataset = TextDataset(corpus_path, context_size)
    vocab_size = len(full_dataset.vocab)
    
    # Calculate split sizes
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    if model_type == 'f':
        model = FFNNLM(vocab_size, embed_dim, hidden_dim, context_size-1)
    elif model_type == 'r':
        model = RNNLM(vocab_size, embed_dim, hidden_dim)
    elif model_type == 'l':
        model = LSTMLM(vocab_size, embed_dim, hidden_dim)
    
    # Store model parameters
    model_params = {
        'model_type': model_type,
        'vocab_size': vocab_size,
        'embed_dim': embed_dim,
        'hidden_dim': hidden_dim,
        'context_size': context_size-1 if model_type == 'f' else None
    }
    
    # Training setup
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.NLLLoss()
    
    best_val_loss = float('inf')
    best_model_state = None

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for contexts, targets in train_loader:
            optimizer.zero_grad()
            if model_type == 'f':
                log_probs = model(contexts)
            else:
                hidden = model.init_hidden(contexts.size(0))
                log_probs, _ = model(contexts.unsqueeze(1), hidden)
            loss = loss_fn(log_probs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for contexts, targets in val_loader:
                if model_type == 'f':
                    log_probs = model(contexts)
                else:
                    hidden = model.init_hidden(contexts.size(0))
                    log_probs, _ = model(contexts.unsqueeze(1), hidden)
                loss = loss_fn(log_probs, targets)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Load best model before saving
    model.load_state_dict(best_model_state)
    
    # Save checkpoint
    corpus_name = corpus_path.split('/')[-1].split('.')[0]
    torch.save({
        'state_dict': model.state_dict(),
        'vocab': dict(full_dataset.vocab),
        'model_params': model_params,
        'corpus_name': corpus_name,
        'best_val_loss': best_val_loss
    }, f"models/{model_type}_{context_size}_{corpus_name}.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--corpus_path", required=True, help="Path to corpus file")
    parser.add_argument("model_type", choices=['f', 'r', 'l'], help="Model type: f for FFNN, r for RNN, l for LSTM")
    parser.add_argument("-n", "--context_size", type=int, default=3, help="Size of context for model training")
    args = parser.parse_args()
    
    train_model(args.corpus_path, args.model_type, args.context_size)