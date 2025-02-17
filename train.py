import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from model import FFNNLM, RNNLM, LSTMLM
from tokenizer import tokenize
from random import shuffle

class TextDataset(Dataset):
    def __init__(self, corpus_path, context_size=3):
        with open(corpus_path, 'r') as f:
            text = f.read()
        
        # Use tokenizer to get sentences
        self.sentences = tokenize(text)
        
        # Build vocabulary
        self.vocab = defaultdict(lambda: 0)
        word_counts = defaultdict(int)
        
        # Add special tokens
        self.vocab['<unk>'] = 0
        self.vocab['<s>'] = 1
        self.vocab['</s>'] = 2
        
        # Count words from all sentences
        for sentence in self.sentences:
            for word in sentence:
                word_counts[word] += 1
        
        # Assign indices
        for idx, (word, count) in enumerate(word_counts.items(), start=3):
            self.vocab[word] = idx
            
        # Create context-target pairs sentence by sentence with padding
        self.data = []
        self.sentence_boundaries = []  # Store start index of each sentence in data
        
        for sentence in self.sentences:
            # Pad the sentence with (context_size-1) <s> tokens at the beginning
            padded_sentence = ['<s>'] * (context_size - 1) + sentence + ['</s>']
            # Convert sentence to indices
            sent_indices = [self.vocab.get(word, 0) for word in padded_sentence]
            
            self.sentence_boundaries.append(len(self.data))
            
            # Create context-target pairs for this padded sentence
            for i in range(len(sent_indices) - context_size):
                context = sent_indices[i:i+context_size]
                target = sent_indices[i+context_size]
                self.data.append((torch.tensor(context), torch.tensor(target)))
        
        # Add final boundary
        self.sentence_boundaries.append(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Index out of range")
        return self.data[idx]

    def get_sentence_indices(self, sentence_idx):
        """Get all data indices for a given sentence"""
        start = self.sentence_boundaries[sentence_idx]
        end = self.sentence_boundaries[sentence_idx + 1]
        return list(range(start, end))

def calculate_perplexity(model, data_loader, device, model_type):
    model.eval()
    total_nll = 0
    total_words = 0
    
    with torch.no_grad():
        for contexts, targets in data_loader:
            contexts = contexts.to(device)
            targets = targets.to(device)
            
            if model_type == 'f':
                log_probs = model(contexts)
            else:
                hidden = model.init_hidden(contexts.size(0))
                if isinstance(hidden, tuple):
                    hidden = tuple(h.to(device) for h in hidden)
                else:
                    hidden = hidden.to(device)
                log_probs, _ = model(contexts, hidden)
            
            # Sum negative log likelihood
            nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze()
            total_nll += nll.sum().item()
            total_words += targets.size(0)
    
    # Calculate perplexity
    avg_nll = total_nll / total_words
    perplexity = torch.exp(torch.tensor(avg_nll)).item()
    return perplexity

def train_model(corpus_path, model_type="f", context_size=3, embed_dim=100, hidden_dim=128, epochs=10, val_split=0.1, dropout=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize dataset
    full_dataset = TextDataset(corpus_path, context_size)
    vocab_size = len(full_dataset.vocab)
    
    # Get sentence indices for splitting
    num_sentences = len(full_dataset.sentences)
    sentence_indices = list(range(num_sentences))
    shuffle(sentence_indices)  # Shuffle sentences
    
    # Split sentences
    test_sentences = sentence_indices[:1000]  # First 1000 sentences for test
    remaining_sentences = sentence_indices[1000:]
    val_size = int(len(remaining_sentences) * val_split)
    val_sentences = remaining_sentences[:val_size]
    train_sentences = remaining_sentences[val_size:]
    
    # Get data indices for each split
    train_indices = []
    val_indices = []
    test_indices = []
    
    for sent_idx in train_sentences:
        train_indices.extend(full_dataset.get_sentence_indices(sent_idx))
    for sent_idx in val_sentences:
        val_indices.extend(full_dataset.get_sentence_indices(sent_idx))
    for sent_idx in test_sentences:
        test_indices.extend(full_dataset.get_sentence_indices(sent_idx))
    
    # Create dataset splits
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    print(f"Dataset splits (sentences): Train={len(train_sentences)}, Val={len(val_sentences)}, Test={len(test_sentences)}")
    # print(f"Dataset splits (samples): Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    if model_type == 'f':
        model = FFNNLM(vocab_size, embed_dim, hidden_dim, context_size, dropout)
    elif model_type == 'r':
        model = RNNLM(vocab_size, embed_dim, hidden_dim, dropout)
    elif model_type == 'l':
        model = LSTMLM(vocab_size, embed_dim, hidden_dim, dropout)
    
    # Store model parameters
    model_params = {
        'model_type': model_type,
        'vocab_size': vocab_size,
        'embed_dim': embed_dim,
        'hidden_dim': hidden_dim,
        'context_size': context_size if model_type == 'f' else None,
        'weight_decay': 0.0001
    }

    # Add weight decay to the optimizer
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=model_params['weight_decay'])
    loss_fn = nn.NLLLoss()
    
    best_val_loss = float('inf')
    best_model_state = None

    # Move model to device
    model = model.to(device)
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for contexts, targets in train_loader:
            # Move data to device
            contexts = contexts.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            if model_type == 'f':
                log_probs = model(contexts)
            else:
                hidden = model.init_hidden(contexts.size(0))
                # Move hidden state to device for RNN/LSTM
                if isinstance(hidden, tuple):
                    hidden = tuple(h.to(device) for h in hidden)
                else:
                    hidden = hidden.to(device)
                log_probs, _ = model(contexts, hidden)
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
                # Move data to device
                contexts = contexts.to(device)
                targets = targets.to(device)
                
                if model_type == 'f':
                    log_probs = model(contexts)
                else:
                    hidden = model.init_hidden(contexts.size(0))
                    # Move hidden state to device for RNN/LSTM
                    if isinstance(hidden, tuple):
                        hidden = tuple(h.to(device) for h in hidden)
                    else:
                        hidden = hidden.to(device)
                    log_probs, _ = model(contexts, hidden)
                loss = loss_fn(log_probs, targets)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        
        # Calculate perplexity for both validation and training sets
        train_perplexity = calculate_perplexity(model, train_loader, device, model_type)
        val_perplexity = calculate_perplexity(model, val_loader, device, model_type)
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Perplexity: {train_perplexity:.2f}, Val Perplexity: {val_perplexity:.2f}")

    # Load best model before final evaluation
    model.load_state_dict(best_model_state)
    
    # Calculate final test perplexity
    test_perplexity = calculate_perplexity(model, test_loader, device, model_type)
    print(f"\nFinal Test Perplexity: {test_perplexity:.2f}")
    
    # Save checkpoint with additional metrics
    corpus_name = corpus_path.split('/')[-1].split('.')[0]
    torch.save({
        'state_dict': model.state_dict(),
        'vocab': dict(full_dataset.vocab),
        'model_params': model_params,
        'corpus_name': corpus_name,
        'best_val_loss': best_val_loss,
        'test_perplexity': test_perplexity,
        'hidden_dim': hidden_dim if model_type != 'f' else None,
        'embed_dim': embed_dim if model_type != 'f' else None,
        'dropout': dropout
    }, f"models/{model_type}{'_'+str(context_size) if model_type=='f' else ''}_{corpus_name}.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--corpus_path", required=True, help="Path to corpus file")
    parser.add_argument("model_type", choices=['f', 'r', 'l'], help="Model type: f for FFNN, r for RNN, l for LSTM")
    parser.add_argument("-n", "--context_size", type=int, default=3, help="Size of context for model training")
    parser.add_argument("-e", "--embed_dim", type=int, default=100, help="Embedding dimension")
    parser.add_argument("-hd", "--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("-ep", "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-d", "--dropout", type=float, default=0.5, help="Dropout rate")
    args = parser.parse_args()
    
    train_model(args.corpus_path, args.model_type, args.context_size, args.embed_dim, args.hidden_dim, args.epochs, args.dropout)