import os
from flask import *
from app.home_views import home
from app.login_views import login
from app.ico_views import ico
from app.models import db


'''
functions.py中配置重要的内容
'''


def create_app():
    # 定义系统路径的变量
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 定义静态文件的变量
    static_dir = os.path.join(BASE_DIR, 'static')
    # 定义模板文件路径
    templates_dir = os.path.join(BASE_DIR, 'templates')
    # 初始化app和manage.py文件关联
    app = Flask(__name__, static_folder=static_dir,
                template_folder=templates_dir)

    # 注册蓝图
    app.register_blueprint(blueprint=home, url_prefix='/home')
    app.register_blueprint(blueprint=login, url_prefix='/index')
    app.register_blueprint(blueprint=ico, url_prefix='/')

    # 配置mysql数据库
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:suda10130212@127.0.0.1:3306/backdoor_eval'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ECHO'] = True

    # 配置session密钥
    app.config['SECRET_KEY'] = 'secret_key'
    app.config['JSON_AS_ASCII'] = False

    # 初始化db
    db.init_app(app=app)

    return app
