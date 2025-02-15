## Execution Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train a model:
```bash
python train.py --corpus ./pride.txt --model_type ffnn --context_size 3
```

3. Generate predictions:
```bash
python generator.py -f ./model.pt 3
Input sentence: It is a truth universally acknowledged
Output: that 0.45 which 0.30 when 0.15
```

Pretrained models included:
- ffnn_n3_pride.pt: FFNN (n=3) trained on Pride and Prejudice
- lstm_ulysses.pt: LSTM trained on Ulysses