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
        sentences = tokenize(text)
        
        # Build vocabulary from all sentences
        self.vocab = defaultdict(lambda: 0)  # Default to <unk>
        word_counts = defaultdict(int)
        
        # Add special tokens
        self.vocab['<unk>'] = 0
        self.vocab['<s>'] = 1
        self.vocab['</s>'] = 2
        
        # Count words
        for sentence in sentences:
            for word in sentence:
                word_counts[word] += 1
        
        # Assign indices (starting from 3 as 0,1,2 are reserved)
        for idx, (word, count) in enumerate(word_counts.items(), start=3):
            self.vocab[word] = idx
            
        # Process each sentence into context-target pairs
        self.data = []
        for sentence in sentences:
            # Add sentence boundaries for FFNN
            padded_sentence = ['<s>'] * (context_size-1) + sentence + ['</s>']
            
            # Convert sentence to indices
            sent_indices = [self.vocab.get(word, 0) for word in padded_sentence]
            
            # Create context-target pairs
            for i in range(len(sent_indices)-context_size+1):
                context = sent_indices[i:i+context_size-1]
                target = sent_indices[i+context_size-1]
                self.data.append((torch.tensor(context), torch.tensor(target)))
        
        # Store original sentences for split purposes
        shuffle(sentences)
        self.sentences = sentences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

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

def train_model(corpus_path, model_type="f", context_size=3, embed_dim=100, hidden_dim=128, epochs=10, val_split=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize dataset
    full_dataset = TextDataset(corpus_path, context_size)
    vocab_size = len(full_dataset.vocab)
    
    # Get total number of sentences
    total_sentences = len(full_dataset.sentences)
    
    # Calculate split sizes for sentences
    test_size = min(1000, total_sentences // 3)  # 1000 sentences or 1/3 if less
    remaining_size = total_sentences - test_size
    val_size = int(remaining_size * val_split)
    train_size = remaining_size - val_size
    
    # Create sentence indices for splits
    all_indices = list(range(total_sentences))
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size+val_size]
    test_indices = all_indices[-test_size:]
    
    # Function to get data points corresponding to sentence indices
    def get_sentence_data_points(indices):
        data_points = []
        current_idx = 0
        for i, sentence in enumerate(full_dataset.sentences):
            sentence_length = len(sentence) + context_size  # Add context_size for padding
            if i in indices:
                start_idx = current_idx
                end_idx = current_idx + sentence_length - context_size + 1
                data_points.extend(range(start_idx, end_idx))
            current_idx += sentence_length - context_size + 1
        return data_points
    
    # Create subset datasets based on sentence indices
    train_dataset = torch.utils.data.Subset(full_dataset, get_sentence_data_points(train_indices))
    val_dataset = torch.utils.data.Subset(full_dataset, get_sentence_data_points(val_indices))
    test_dataset = torch.utils.data.Subset(full_dataset, get_sentence_data_points(test_indices))
    
    print(f"Dataset splits (sentences): Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
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
        'context_size': context_size-1 if model_type == 'f' else None,
        'weight_decay': 0.01
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
        'embed_dim': embed_dim if model_type != 'f' else None
    }, f"models/{model_type}_{context_size}_{corpus_name}.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--corpus_path", required=True, help="Path to corpus file")
    parser.add_argument("model_type", choices=['f', 'r', 'l'], help="Model type: f for FFNN, r for RNN, l for LSTM")
    parser.add_argument("-n", "--context_size", type=int, default=3, help="Size of context for model training")
    parser.add_argument("-e", "--embed_dim", type=int, default=100, help="Embedding dimension")
    parser.add_argument("-d", "--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("-ep", "--epochs", type=int, default=10, help="Number of epochs")
    args = parser.parse_args()
    
    train_model(args.corpus_path, args.model_type, args.context_size, args.embed_dim, args.hidden_dim, args.epochs)