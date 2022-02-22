"""Classes and functions for handling user files"""
import contextlib
import logging
import ttkthemes
import tkinter.messagebox
import json
import sys
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

    def is_valid(self) -> bool:
      """Check that path is not :data:`None` or empty"""
      return self.userdir.is_valid() and self.subpath is not None and len(str(self.subpath)) > 0

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

  def is_valid(self) -> bool:
    """Check that path is not :data:`None` or empty"""
    return self.path is not None and len(str(self.path)) > 0

  def user_file(self, subpath):
    """Get a user file utility object

    Args:
      subpath: Path of the file in the directory"""
    return self.UserFile(userdir=self, subpath=subpath)


class UserTtkTheme:
  @staticmethod
  def is_valid(theme: str, log: bool = False, messagebox: bool = False) -> bool:
    b = theme in ttkthemes.THEMES
    if not b and (log or messagebox):
      m = f"Unsupported theme: '{theme}'. Supported themes are: "
      m += ", ".join(f"'{t}'" for t in ttkthemes.THEMES)
      if log:
        logging.warning(m)
      if messagebox:
        tkinter.messagebox.showwarning("Unsupported theme", m)
    return b

  @staticmethod
  def default() -> str:
    if sys.platform == "linux":
      return "radiance"
    else:
      return "arc"

  def __init__(self, file: UserDir.UserFile):
    self.file = file

  def get(self, k: str = "gui_theme") -> str:
    if self.file.is_valid() and self.file.exists():
      s = self.file.load_json()
      if k in s:
        return s[k]
    return self.default()
