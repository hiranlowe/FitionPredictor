from ast import Str
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from database import Base


class Activity(Base):
    __tablename__ = "activities"

    id = Column(Integer, primary_key=True, index=True)
    sentiment = Column(String)
    sentiment_text = Column(String)
    probability = Column(String)
    current_time = Column(String)


