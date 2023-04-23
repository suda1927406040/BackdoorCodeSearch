from flask_sqlalchemy import SQLAlchemy
# from sqlalchemy import text

db = SQLAlchemy()


# class Parameters(db.Model):
#     def __init__(self) -> None:
#         pass


class User(db.Model):
    # 用户表
    # 定义表名
    __tablename__ = 'ymz_users'
    u_id = db.Column(db.Integer(), autoincrement=True, primary_key=True)
    u_name = db.Column(db.String(6), unique=True)
    pwd = db.Column(db.String(16))

    # 定义保存数据的方法，方便后面使用
    def save(self):
        db.session.add(self)
        db.session.commit()


# class Attack(db.Model):
#     # 用户表
#     # 定义表名
#     __tablename__ = 'ymz_models'
#     m_id = db.Column(db.Integer(), autoincrement=True, primary_key=True)
#     m_name = db.Column(db.String(6))
#     target = db.Column(db.String(16))
#     trigger = db.Column(db.String(255))
#
#     # 定义保存数据的方法，方便后面使用
#     def save(self):
#         db.session.add(self)
#         db.session.commit()


# class Ranking(db.Model):
#
#     # 评估输入表
#     # 定义表名
#     __tablename__ = 'ymz_rankings'
#     model_id = db.Column(db.Integer(), autoincrement=True, primary_key=True)
#     model_name = db.Column(db.String(16))
#     score = db.Column(db.Float)
#
#     # 外键
#     user_id = db.Column(db.Integer, db.ForeignKey('ymz_users.user_id'))
#     user = db.relationship('User', backref='user', foreign_keys=user_id)
#
#     def save(self):
#         db.session.add(self)
#         db.session.commit()
