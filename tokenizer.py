import re
def tokenize(text):

    text = re.sub(r'http\S+', '<URL>', text)
    text = re.sub(r'www\S+', '<URL>', text)
    text = re.sub(r'[A-Za-z0-9._%+-]+@[A-za-z0-9.-]+\.[a-z]{2,}', '<MAILID>', text)

    text = re.sub(r'[^\@\#\.\w\?\!\s:-]', '', text)
    text = re.sub(f'-', ' ', text)
    text = re.sub(r'_', ' ', text)

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n*', '', text)
    text = re.sub(r'\.+', '.', text)

    abbreviations = re.findall(r'\b([A-Z]([a-z]){,2}\.)', text)
    if(abbreviations):
        abbreviations_set = set((list(zip(*abbreviations))[0]))

        for word in abbreviations_set:
            pattern = r'\b' + re.escape(word)
            text = re.sub(pattern, word.strip('.'), text)

    text = re.sub(r'#\w+\b', '<HASHTAG>', text)

    text = re.sub(r'@\w+\b', '<MENTION>', text)

    text = re.sub(r'\b\d+\b', '<NUM>', text)

    sentences = re.split(r'[.!?:]+', text)
    
    sentences = [sentence.strip() for sentence in sentences]

    dummy = []
    for sentence in sentences:
        current_word = ''
        tokens_in_sentence = []
        for char in sentence:
            if char != ' ':
                current_word += char
            elif current_word:
                tokens_in_sentence.append(current_word)
                current_word = ''
        if current_word:
            tokens_in_sentence.append(current_word)
        dummy.append(tokens_in_sentence)

    sentences = dummy

    sentences = [[word for word in sentence if word != ''] for sentence in sentences]
    sentences = [sentence for sentence in sentences if sentence]

    return sentences