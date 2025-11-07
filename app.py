from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.post('/hit')
def hit():
    data = request.get_json(force=True)
    print("HIT:", data)  # shows in your terminal
    return jsonify({"ok": True})
    
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/play/<game>')
def play(game):
    # restrict to known games
    if game not in ('around', 'x01'):
        game = 'around'
    start = int(request.args.get('start', 501))
    double_out = request.args.get('double_out', '1') == '1'
    return render_template('play.html', game=game, start=start, double_out=double_out)

if __name__ == '__main__':
    app.run(debug=True)