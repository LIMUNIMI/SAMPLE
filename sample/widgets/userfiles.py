"""Classes and functions for handling user files"""
import contextlib
import functools
import json
import os


class UserDir:
  """Utility class for the user files directory
  
  Args:
    path: Directory path
    in_home (bool): If :data:`True`, the :data:`path` is a 
      subpath of the user home directory"""
  class UserFile:
    """Utility class for user files

    Args:
      userdir (UserDir): User directory object
      subpath: Path of the file inside the directory"""
    def __init__(self, userdir: "UserDir", subpath) -> None:
      self.subpath = subpath
      self.userdir = userdir

    @property
    def path(self) -> str:
      """Full path of the file"""
      return os.path.join(self.userdir.path, self.subpath)

    @contextlib.contextmanager
    def open(self, encoding = "utf-8", **kwargs):
      """Helper context manager to interface with :func:`os.open`"""
      os.makedirs(os.path.dirname(self.path), exist_ok=True)
      with open(self.path, encoding=encoding, **kwargs) as f:
        yield f

    def exists(self) -> bool:
      """Test existance of file"""
      return os.path.isfile(self.path)

    def save_json(self,data, **kwargs):
      with self.open(mode="w") as f:
        json.dump(data, f, **kwargs)

    def load_json(self, **kwargs):
      with self.open(mode="r") as f:
        return json.load(f, **kwargs)

  def __init__(self, path, in_home: bool = False):
    if in_home:
      path = os.path.expanduser(os.path.join("~", path))
    self.path = path

  def user_file(self, subpath):
    """Get a user file utility object

    Args:
      subpath: Path of the file in the directory"""
    return self.UserFile(userdir=self, subpath=subpath)
