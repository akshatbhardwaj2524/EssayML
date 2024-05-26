import re
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class SimpleSpellingGrammarChecker:
    def __init__(self):
        pass

    def check_spelling(self, text):
        # Split text into words and check spelling
        words = nltk.word_tokenize(text)
        misspelled_words = [(word, "Spelling") for word in words if not self.is_spelled_correctly(word)]
        return misspelled_words

    def check_grammar(self, text):
        # Check for common grammar mistakes
        errors = []
        # Example: check for repeated words
        repeated_words = re.findall(r'\b(\w+)\s+\1\b', text)
        for word in repeated_words:
            errors.append((f"Repeated word: {word}", "Grammar"))
        
        # Use POS tagging to identify grammar errors
        tagged_words = nltk.pos_tag(nltk.word_tokenize(text))
        for word, tag in tagged_words:
            if tag.startswith('V'):  # Check for verbs
                errors.append((f"Incorrect usage of verb: {word}", "Grammar"))
        return errors

    def is_spelled_correctly(self, word):
        # Simple spelling check using nltk
        return word.lower() in set(nltk.corpus.words.words())

if __name__ == "__main__":
    checker = SimpleSpellingGrammarChecker()
    text = "Hello, hw are you? I am fine. This is an example of spellling  grammer check."
    
    # Check spelling
    misspelled_words = checker.check_spelling(text)
    if misspelled_words:
        print("Errors found:")
        for error, error_type in misspelled_words:
            print(f"{error_type} error: {error}")
    else:
        print("No spelling errors found.")
    
    # Check grammar
    grammar_errors = checker.check_grammar(text)
    if grammar_errors:
        print("\nErrors found:")
        for error, error_type in grammar_errors:
            print(f"{error_type} error: {error}")
    else:
        print("\nNo grammar errors found.")
