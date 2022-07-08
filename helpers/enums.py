__all__ = ["SimpleEnum"]


class EnumMeta(type):
  def __getattribute__(self, name: str) -> str:
    annotations = object.__getattribute__(self, "__annotations__")
    if name in annotations:
      return name
    class_name = object.__getattribute__(self, "__class__")
    raise AttributeError(f"Object {class_name} has no attribute {name!r}.")


class SimpleEnum(metaclass=EnumMeta):
  """
  Similar to Enum, except that instead of calling `SimpleEnum.MyField.name`, you simply call `SimpleEnum.MyField`
  to obtain the name of the field.
  """
  pass
