# server.py
from flask import Flask, render_template
import os

app = Flask(__name__, static_url_path = "/home/tatsuyasuzuki/sal_server/static", static_folder = "static")
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = os.path.join("/home/tatsuyasuzuki/sal_server/static")
app.config['TEMPLATES_AUTO_RELOAD'] = True      
app.jinja_env.auto_reload = True
print(__name__)

@app.route('/', methods=['POST', 'GET', 'PUT'])
def index():
    path2img = os.path.join(app.config['UPLOAD_FOLDER'], "output.png") 
    print(path2img)
    return render_template('index.html', sal_img=path2img)

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response

if __name__ == '__main__':
    app.debug = True
    app.run(host='192.168.11.61', port=10030, debug=True)

