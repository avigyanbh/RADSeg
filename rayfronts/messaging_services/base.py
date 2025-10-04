import abc
import threading

class MessagingService:

  @abc.abstractmethod
  def text_query_handler(self, s):
    pass

  @abc.abstractmethod
  def broadcast_gps_message(self, lat, long):
    pass

  @abc.abstractmethod
  def join(self, timeout = None):
    pass

  @abc.abstractmethod
  def shutdown(self):
    pass
