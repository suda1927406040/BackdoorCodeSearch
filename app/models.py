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
    user_id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    username = db.Column(db.String(6))
    password = db.Column(db.String(16))

    # 定义保存数据的方法，方便后面使用
    def save(self):
        db.session.add(self)
        db.session.commit()


# class Ranking(db.Model):

#     # 评估输入表
#     # 定义表名
#     __tablename__ = 'ymz_rankings'
#     user_id =
#     model_id = db.Column()
#     score = db.Column()

#     def save(self):
#         db.session.add(self)
#         db.session.commit()
