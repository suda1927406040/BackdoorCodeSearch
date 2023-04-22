# from flask import Flask, request, render_template, Blueprint
from utils.functions import create_app
from utils.comment import scheduler

app = create_app()
scheduler.init_app(app)
scheduler.start()


# @app.route('/')
# def index():
#     return '请进入/home/index'


if __name__ == '__main__':
    app.run(debug=True, port=5000)
