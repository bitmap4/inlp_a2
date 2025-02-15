import torch
import argparse
from model import FFNNLM, RNNLM, LSTMLM

def load_model(model_path):
    """Load model checkpoint and reconstruct model architecture"""
    checkpoint = torch.load(model_path)
    model_params = checkpoint['model_params']
    
    # Instantiate correct model type
    if model_params['model_type'] == 'f':
        model = FFNNLM(
            vocab_size=model_params['vocab_size'],
            embed_dim=model_params['embed_dim'],
            hidden_dim=model_params['hidden_dim'],
            context_size=model_params['context_size']
        )
    elif model_params['model_type'] == 'r':
        model = RNNLM(
            vocab_size=model_params['vocab_size'],
            embed_dim=model_params['embed_dim'],
            hidden_dim=model_params['hidden_dim']
        )
    elif model_params['model_type'] == 'l':
        model = LSTMLM(
            vocab_size=model_params['vocab_size'],
            embed_dim=model_params['embed_dim'],
            hidden_dim=model_params['hidden_dim']
        )
    
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # Create reverse vocabulary (index -> word)
    vocab = {v: k for k, v in checkpoint['vocab'].items()}
    return model, vocab, model_params

def preprocess_input(sentence, vocab, model_params):
    """Convert input sentence to model-ready tensor"""
    # Tokenize and convert to indices
    tokens = sentence.strip().split()
    word_to_idx = {k: v for v, k in vocab.items()}  # Reverse lookup
    unk_idx = word_to_idx.get('<unk>', 0)
    
    indices = [word_to_idx.get(token.lower(), unk_idx) for token in tokens]
    
    # Model-specific processing
    if model_params['model_type'] == 'f':
        # FFNN: Use last n-1 words
        context_size = model_params['context_size']
        if len(indices) >= context_size:
            indices = indices[-context_size:]
        else:
            indices = [unk_idx] * (context_size - len(indices)) + indices
        
        return torch.tensor(indices).unsqueeze(0)  # Add batch dimension
    
    else:  # RNN/LSTM
        # Process full sequence
        return torch.tensor(indices).unsqueeze(0)  # Add batch dimension

def predict_next_words(model, model_params, input_tensor, k=3):
    """Run model inference and return top k predictions"""
    with torch.no_grad():
        if model_params['model_type'] == 'f':
            log_probs = model(input_tensor)
        else:
            # Initialize hidden state
            batch_size = input_tensor.size(0)
            if isinstance(model, RNNLM):
                hidden = torch.zeros(1, batch_size, model.hidden_dim)
            else:  # LSTM
                hidden = (
                    torch.zeros(1, batch_size, model.hidden_dim),
                    torch.zeros(1, batch_size, model.hidden_dim)
                )
            log_probs, _ = model(input_tensor, hidden)
        
    probs = torch.exp(log_probs)
    top_probs, top_indices = probs.topk(k)
    return top_indices.squeeze().tolist(), top_probs.squeeze().tolist()

def main():
    parser = argparse.ArgumentParser(description='Next Word Prediction Generator')
    parser.add_argument('model_path', help='Path to pretrained model (.pt file)')
    parser.add_argument('k', type=int, help='Number of predictions to return')
    args = parser.parse_args()

    # Load model and vocabulary
    model, vocab, model_params = load_model(args.model_path)
    
    # Get user input
    sentence = input("Input sentence: ").strip()
    
    # Preprocess input
    input_tensor = preprocess_input(sentence, vocab, model_params)
    
    # Get predictions
    indices, probs = predict_next_words(model, model_params, input_tensor, args.k)
    
    # Convert indices to words
    predictions = [(vocab[idx], prob) for idx, prob in zip(indices, probs)]
    
    # Format output
    print("Output:", " ".join(f"{word} {prob}" for word, prob in predictions))

if __name__ == "__main__":
    main()