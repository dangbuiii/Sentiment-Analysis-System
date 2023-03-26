from flask import Flask, render_template, request, redirect, url_for, views
import  re
import Test2

app = Flask(__name__)

@app.route("/", methods=['GET'])
def test():
    return render_template('index.html')


@app.route("/app", methods=['POST'])
def appreciate():
    keyword = request.form['keyword']
    #xử lý ngoại lệ
    match = re.search(r'(.*) không thể không thích', keyword)
    if match != None:
        result = 'positive'
    else:
        result = str(Test2.testComment(keyword))
    return result


if __name__ == "__main__":
    app.run()
