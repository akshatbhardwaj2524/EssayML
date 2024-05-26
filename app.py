from flask import Flask, request, render_template
from MODEL import SimpleSpellingGrammarChecker

app = Flask(__name__)
simple_spelling_grammar_checker = SimpleSpellingGrammarChecker()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/spell', methods=['POST'])
def spell():
    if request.method == 'POST':
        text = request.form['text']
        corrected_text = simple_spelling_grammar_checker.check_spelling(text)
        corrected_grammar = simple_spelling_grammar_checker.check_grammar(text)
        return render_template('index.html', corrected_text=corrected_text, corrected_grammar=corrected_grammar)

@app.route('/grammar', methods=['POST'])
def grammar():
    if request.method == 'POST':
        file_content = request.form['file']
        corrected_grammar = simple_spelling_grammar_checker.check_grammar(file_content)
        return render_template('index.html', corrected_grammar=corrected_grammar)

if __name__ == "__main__":
    app.run(debug=True)
