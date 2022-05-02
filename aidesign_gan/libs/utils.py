"""Utilities."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

# dotdict imports
from collections import abc

# timedinput imports
import asyncio
import sys
import threading

# jsonrw imports
import json

# textrw imports
import typing

# randbool imports
import random

# batchlog imports
# import typing

# dotdict aliases

_Iterable = abc.Iterable
_Mapping = abc.Mapping

# timedinput aliases
_create_subprocess_exec = asyncio.create_subprocess_exec
_executable = sys.executable
_PIPE = asyncio.subprocess.PIPE
_run = asyncio.run
_stdin = sys.stdin
_Thread = threading.Thread

# jsonrw aliases
_jsondump = json.dump
_jsondumps = json.dumps
_jsonload = json.load
_jsonloads = json.loads
_NoneType = type(None)

# textrw aliases
_IO = typing.IO

# randbool aliases
_randint = random.randint

# batchlog aliases
# _IO = typing.IO


class DotDict(dict):
    """Dot dictionary, API version 2.

    NOTE: Not API-compatible with the version 1 DotDict.

    A dictionary whose items can ...
        ... be accessed with the "dict.key" syntax, with some exceptions; and,
        ... be accessed with the "dict[key]" syntax, similar to the items in a Python built-in dict.

    A child class of the Python built-in dict, with the type dict[str, typing.Any].
    Recursively compatible with the Python built-in dict in nested structures.

    "dict.key" syntax exception 1: If a key is included in the dir(dict()) attribute list, it is read-only and
        evaluated to the respective dict attribute.

    "dict.key" syntax exception 2: A key is protected -- read-only, and evaluated to the respective DotDict attribute,
        if it is ...
        ... private -- in the form of "_(.*)"; or,
        ... magic -- in the form of "__(.*)__"; or,
        ... has 4 trailing underscores -- in the form of "(.*)____".

    However, the "dict[key]" syntax has no such exceptions.

    If a key is of the above "dict.key" exceptional forms, when accessed with the "dict[key]" syntax, there are no
        restrictions similar to the above ones; the key will be ...
        ... writable, as in a Python built-in dict; and,
        ... with the "dict.key" initial values if "dict.key" exists; and,
        ... with no initial value if "dict.key" does not exist.
    """

    # Part of LYC-PythonUtils
    # Copyright 2022 Yucheng Liu. GNU LGPL3 license.
    # GNU LGPL3 license copy: https://www.gnu.org/licenses/lgpl-3.0.txt
    # GNU LGPL3 is based on GNU GPL3, GNU GPL3 copy: https://www.gnu.org/licenses/gpl-3.0.txt

    dir_dict_excs____ = dir(dict())
    """The dir(dict()) exceptional names."""

    @classmethod
    def from_dict____(cls, dict_):
        """Builds and returns a DotDict from a Python built-in dict or Mapping.

        Args:
            dict_: a dict or Mapping

        Returns:
            result: the resulting DotDict
        """
        # print(f"fromdict____ dic: {dic}")  # Debug
        result = DotDict()

        for key in dict_:
            result.set_attr____(key, dict_[key])

        return result

    @classmethod
    def is_exc_key____(cls, key):
        """Finds if a key is exceptional.

        If a key is exceptional, when accessed with the "dict.key" syntax, it is read-only and evaluated to the
            respective dict or DotDict attribute.

        Args:
            key: a key

        Returns:
            result: whether the key is exceptional
        """
        # print(f"is_exc____ key: {key}")  # Debug
        key = str(key)

        # See whether the key is in the dir(dict()) attribute list
        result = key in cls.dir_dict_excs____

        # See whether the key ...
        #   ... is private; or,
        #   ... is magic; or,
        #   ... has 4 trailing underscores
        result = result or \
            key[:1] == "_" or \
            (len(key) >= 4 and key[:2] == "__" and key[-2:] == "__") or \
            key[-4:] == "____"

        return result

    def _mand_init(self, *args, **kwargs):
        """Inits self with the given args and kwargs.

        This is a mandatory method to call in the initialization process.

        Args:
            *args: the variable arguments
            **kwargs: the keyword arguments
        """
        self_type = type(self)

        # Set the inherited keys from the custom class level
        if self_type is not DotDict and issubclass(self_type, DotDict):
            class_dict = self_type.__dict__

            for key in class_dict:
                key = str(key)
                # print(f"_dot_dict_init class_dict {key}: {class_dict[key]}")  # Debug

                if not self_type.is_exc_key____(key):
                    val = class_dict[key]
                    self.set_attr____(key, val)
            # end for
        # end if

        # Set keys with the key names from the variable arguments
        for arg in args:
            arg = str(arg)
            # print(f"_dot_dict_init *args arg: {arg}")  # Debug

            if not self_type.is_exc_key____(key):
                self.set_attr____(arg, None)
        # end for

        # Set pairs with the key names and values from the keyword arguments
        for kw in kwargs:
            kw = str(kw)
            # print(f"_dot_dict_init **kwargs kw: {kw}  arg: {kwargs[kw]}")  # Debug

            if not self_type.is_exc_key____(key):
                arg = kwargs[kw]
                self.set_attr____(kw, arg)
        # end for

    def _empty_init(self):
        """Inits self."""
        super().__init__()
        self._mand_init()

    def _map_init(self, map_):
        """Inits self with a map.

        Args:
            map_: a map
        """
        super().__init__(map_)
        self._mand_init()

    def _iter_init(self, iter_):
        """Inits self with an iterable.

        Args:
            iter_: an iterable
        """
        super().__init__(iter_)
        self._mand_init()

    def _kwargs_init(self, **kwargs):
        """Inits self with the given keyword arguments.

        Args:
            **kwargs: the keyword arguments
        """
        super().__init__(**kwargs)
        self._mand_init()

    def _dot_dict_init(self, *args, **kwargs):
        """Inits self with the given args and kwargs.

        Args:
            *args: the variable arguments
            **kwargs: the keyword arguments
        """
        super().__init__()
        self._mand_init(*args, **kwargs)

    # Magic functions

    def __init__(self, *args, **kwargs):
        """Inits self with the given args and kwargs.

        Compatible with the following dict initialization methods:
            dict(); and,
            dict(mapping); and,
            dict(iterable); and,
            dict(**kwargs).

        If the given args and kwargs do not fall into the above categories, this method will use the DotDict
            initialization method to complete the initialization.

        Args:
            *args: the variable arguments
            **kwargs: the keyword arguments
        """
        args_len = len(args)
        kwargs_len = len(kwargs)

        empty_args = args_len <= 0
        empty_kwargs = kwargs_len <= 0

        single_arg = args_len == 1 and empty_kwargs
        only_kwargs = empty_args and kwargs_len > 0

        if empty_args and empty_kwargs:
            self._empty_init()
        elif single_arg:
            the_arg = args[0]

            if isinstance(the_arg, _Mapping):
                self._map_init(the_arg)
            elif isinstance(the_arg, _Iterable):
                self._iter_init(the_arg)
            else:
                self._dot_dict_init(*args, **kwargs)
            # end if
        elif only_kwargs:
            self._kwargs_init(**kwargs)
        else:
            self._dot_dict_init(*args, **kwargs)
        # end if

    def __getitem__(self, key):
        """Gets an item with the given key.

        Args:
            key: an item key

        Returns:
            _: the item value
        """
        return self.get_item____(key)

    def __setitem__(self, key, val):
        """Sets an item with the given key to the given value.

        Args:
            key: an item key
            val: an item value

        Returns:
            _: the item value after the setting
        """
        return self.set_item____(key, val)

    def __delitem__(self, key):
        """Deletes an item with the given key.

        Args:
            key: an item key

        Returns:
            _: the item value after the deletion
        """
        return self.del_item____(key)

    def __getattr__(self, name):
        """Gets an attribute with the given name.

        Args:
            name: an attribute name

        Returns:
            _: the attribute value
        """
        return self.get_attr____(name)

    def __setattr__(self, name, val):
        """Sets an attribute with the given name to the given value.

        Args:
            name: an attribute name
            val: an attribute value

        Returns:
            _: the attribute value after the setting
        """
        return self.set_attr____(name, val)

    def __delattr__(self, name):
        """Deletes an attribute with the given name.

        Args:
            name: an attribute name

        Returns:
            _: the attribute value after deletion
        """
        return self.del_attr____(name)

    def __repr__(self):
        """Finds the Python representation of self.

        This makes eval(repr(self)) == self.

        Returns:
            _: self's Python representation
        """
        return self.repr____()

    def __str__(self):
        """Finds the string representation of self.

        Returns:
            _: self's string representation
        """
        return self.str____()

    def __getstate__(self):
        """Finds the serializable form of self.

        Returns:
            _: self's serialization
        """
        return self.to_dict____()

    def __setstate__(self, state):
        """Sets self's internal to the given state.

        Args:
            state: a serialization of self's certain state
        """
        self.__init__(**state)

    # End of magic functions
    # Attribute functions

    def get_attr____(self, name):
        """Gets an attribute with the given name.

        Args:
            name: an attribute name

        Returns:
            val: the attribute value

        Raises:
            AttributeError: if self and self.__dir__() both have no such an attribute
        """
        name = str(name)

        self_dir = self.__dir__()
        name_in_self_dir = name in self_dir
        name_in_self = name in self

        if name_in_self_dir:
            val = super().__getattribute__(name)
        elif name_in_self:
            val = super().__getitem__(name)
        else:
            raise AttributeError(f"self and self.__dir__() both have no attribute called: {name}")
        # end if

        return val

    def set_attr____(self, name, val):
        """Sets an attribute with the given name to the given value.

        If the name is an exceptional key name, this method protects the corresponding attribute from the changes.

        When setting the attribute values, all Python built-in dict values are converted to their corresponding DotDict
            values recursively.

        Args:
            name: an attribute name
            val: an attribute value

        Returns:
            val: The attribute value. None if:
                the original value is None; or,
                the method trys to write a protected but unbounded attribute.
        """
        name = str(name)

        self_type = type(self)
        name_is_exc = self_type.is_exc_key____(name)

        self_dir = self.__dir__()
        name_in_self_dir = name in self_dir
        name_in_self = name in self

        if name_is_exc:
            if name_in_self_dir:
                val = super().__getattribute__(name)
            elif name_in_self:
                val = super().__getitem__(name)
            else:
                val = None
            # end if
        else:
            if (not isinstance(val, DotDict)) and isinstance(val, dict):
                val = DotDict.from_dict____(val)

            if name_in_self_dir:
                super().__setattr__(name, val)

            super().__setitem__(name, val)
        # end if

        return val

    def del_attr____(self, name):
        """Deletes an attribute with the given name.

        If the name is an exceptional key name, this method protects the corresponding attribute from the changes.

        Args:
            name: an attribute name

        Returns:
            val: The attribute value. None if:
                the original value is None; or,
                the deletion is successful.

        Raises:
            AttributeError: if self and self.__dir__() both have no such an attribute
        """
        name = str(name)

        self_type = type(self)
        name_is_exc = self_type.is_exc_key____(name)

        self_dir = self.__dir__()
        name_in_self_dir = name in self_dir
        name_in_self = name in self

        info = f"self and self.__dir__() both have no attribute called: {name}"

        if name_is_exc:
            if name_in_self_dir:
                val = super().__getattribute__(name)
            elif name_in_self:
                val = super().__getitem__(name)
            else:
                raise AttributeError(info)
            # end if
        else:
            if name_in_self_dir:
                super().__delattr__(name)

            if name_in_self:
                super().__delitem__(name)

            if (not name_in_self_dir) and (not name_in_self):
                raise AttributeError(info)

            val = None
        # end if

        return val

    def get_class_attr____(self, name):
        """Gets a class-level (static) attribute with the given name.

        This method also overwrites self.name with the value of type(self).name.

        Args:
            name: an attribute name

        Returns:
            val: the attribute value

        Raises:
            AttributeError: if type(self).__dir__() does not have the attribute
        """
        name = str(name)

        self_type = type(self)
        class_dir = dir(self_type)

        if name not in class_dir:
            raise AttributeError(f"type(self).__dir__() has no attribute called: {name}")

        val = getattr(self_type, name)
        self.set_attr____(name, val)

        return val

    def set_class_attr____(self, name, val):
        """Sets a class-level (static) attribute with the given name to the given value.

        This method:
            sets self.name to the given value; and,
            sets type(self).name to the given value.

        If the name is an exceptional key name, this method protects the corresponding attribute from the changes.

        When setting the attribute values, all Python built-in dict values are converted to their corresponding DotDict
            values recursively.

        Args:
            name: an attribute name
            val: an attribute value

        Returns:
            val: The attribute value. None if:
                the original value is None; or,
                the method trys to write a protected but unbounded attribute.
        """
        name = str(name)

        self_type = type(self)
        name_is_exc = self_type.is_exc_key____(name)

        class_dir = dir(self_type)
        self_dir = self.__dir__()
        name_in_class_dir = name in class_dir
        name_in_self_dir = name in self_dir
        name_in_self = name in self

        if name_is_exc:
            if name_in_class_dir:
                val = getattr(self_type, name)
            elif name_in_self_dir or name_in_self:
                val = self.get_attr____(name)
            else:
                val = None
            # end if
        else:
            if (not isinstance(val, DotDict)) and isinstance(val, dict):
                val = DotDict.from_dict____(val)

            setattr(self_type, name, val)
            self.set_attr____(name, val)
        # end if

        return val

    def del_class_attr____(self, name):
        """Deletes a class-level (static) attribute with the given name.

        This method:
            deletes self.name; and,
            deletes type(self).name.

        If the name is an exceptional key name, this method protects the corresponding attribute from the changes.

        Args:
            name: an attribute name

        Returns:
            val: The attribute value. None if:
                the original value if None; or,
                the deletion is successful.

        Raises:
            AttributeError: if self, self.__dir__(), and type(self).__dir__() all have no such an attribute
        """
        name = str(name)

        self_type = type(self)
        name_is_exc = self_type.is_exc_key____(name)

        class_dir = dir(self_type)
        self_dir = self.__dir__()
        name_in_class_dir = name in class_dir
        name_in_self_dir = name in self_dir
        name_in_self = name in self

        info = f"self, self.__dir__(), and type(self).__dir__() all have no attribute called: {name}"

        if name_is_exc:
            if name_in_class_dir:
                val = getattr(self_type, name)
            elif name_in_self_dir or name_in_self:
                val = self.get_attr____(name)
            else:
                raise AttributeError(info)
            # end if
        else:
            if name_in_class_dir:
                delattr(self_type, name)

            self.del_attr____(name)

            if (not name_in_self) and (not name_in_self_dir) and (not name_in_class_dir):
                raise AttributeError(info)

            val = None
        # end if

        return val

    # End of attribute functions
    # Item functions

    def get_item____(self, key):
        """Gets an item with the given key.

        Args:
            key: an item key

        Returns:
            val: the item value

        Raises:
            KeyError: if self and self.__dir__() both have no such an item
        """
        key = str(key)

        self_dir = self.__dir__()
        key_in_self = key in self
        key_in_self_dir = key in self_dir

        if key_in_self:
            val = super().__getitem__(key)
        elif key_in_self_dir:
            val = super().__getattribute__(key)
        else:
            raise KeyError(f"self and self.__dir__() both have no item with the key: {key}")
        # end if

        return val

    def set_item____(self, key, val):
        """Sets an item with the given key to the given value.

        When setting the item values, all Python built-in dict values are converted to their corresponding DotDict
            values recursively.

        Args:
            key: an item key
            val: an item value

        Returns:
            val: the item value
        """
        key = str(key)

        self_type = type(self)
        key_is_exc = self_type.is_exc_key____(key)

        self_dir = self.__dir__()
        key_in_self_dir = key in self_dir

        if (not isinstance(val, DotDict)) and isinstance(val, dict):
            val = DotDict.from_dict____(val)

        super().__setitem__(key, val)

        if (not key_is_exc) and key_in_self_dir:
            super().__setattr__(key, val)

        return val

    def del_item____(self, key):
        """Deletes an item with the given key.

        Args:
            key: an item key

        Returns:
            val: The item value. None if the deletion is successful.

        Raises:
            KeyError: if self and self.__dir__() both have no such an item
        """
        key = str(key)

        self_type = type(self)
        key_is_exc = self_type.is_exc_key____(key)

        self_dir = self.__dir__()
        key_in_self = key in self
        key_in_self_dir = key in self_dir

        if key_in_self:
            super().__delitem__(key)

        if (not key_is_exc) and key_in_self_dir:
            super().__delattr__(key)

        if (not key_in_self) and (not key_in_self_dir):
            raise KeyError(f"self and self.__dir__() both have no item with the key: {key}")

        val = None
        return val

    def get_class_item____(self, key):
        """Gets a class-level (static) item with the given key.

        Alias of the get_class_attr____ method.

        Args:
            key: an item key

        Returns:
            _: the item value
        """
        return self.get_class_attr____(key)

    def set_class_item____(self, key, val):
        """Sets a class-level (static) item with the given key to the given value.

        Alias of the set_class_attr____ method.

        Args:
            key: an item key
            val: an item value

        Returns:
            _: the item value
        """
        return self.set_class_attr____(key, val)

    def del_class_item____(self, key):
        """Deletes a class-level (static) item with the given key.

        Alias of the del_class_attr____ method.

        Args:
            key: an item key

        Returns:
            _: the item value
        """
        return self.del_class_attr____(key)

    # End of item functions

    def repr____(self):
        """Finds a Python representation of self.

        This makes eval(repr(self)) == self.

        Returns:
            result: the resulting Python representation
        """
        super_repr = super().__repr__()
        result = f"DotDict(**{super_repr})"
        return result

    def str____(self):
        """Finds a string representation of self.

        Returns:
            result: the resulting string representation
        """
        if len(self) <= 0:
            result = ".{}"
            return result

        result = ".{"

        for key in self:
            result += f"{key.__str__()}: {self[key].__str__()}, "

        result = result[:-2]  # Remove the trailing comma and space
        result += "}"

        return result

    def to_dict____(self):
        """Finds a Python built-in dict version of self.

        This method converts all DotDict values to their corresponding Python built-in dict values recursively.

        Returns:
            result: the result dict
        """
        result = {}

        for key in self:
            val = self[key]

            if isinstance(val, DotDict):
                val = val.to_dict____()

            result[key] = val
        # end for

        return result


class TimedInput:
    """Timed input.

    Python "native" and platform independent timed input command prompt.
    """

    # Part of LYC-PythonUtils
    # Copyright 2022 Yucheng Liu. GNU GPL3 license.
    # GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt

    def __init__(self):
        """Inits self with the given args."""
        self._input_str = None

        subproc_code = fr"""

input_str = input()
print(input_str)

"""
        subproc_code = subproc_code.strip()
        subproc_code = subproc_code + "\n"
        self._subproc_code = subproc_code
        self._subproc = None

    async def _async_run_subproc(self):
        self._subproc = await _create_subprocess_exec(
            _executable, "-c", self._subproc_code, stdin=_stdin, stdout=_PIPE
        )

        data = await self._subproc.stdout.readline()
        self._input_str = data.decode("utf-8", "replace").rstrip()
        await self._subproc.wait()

    def _take(self):
        self._subproc = None
        _run(self._async_run_subproc())

    def take(self, timeout=5.0):
        """Takes and returns a string from user input with a given timeout.

        Args:
            timeout: the timeout period length in seconds

        Returns:
            self._input_str: the taken input string, or None if there is a timeout
        """
        timeout = float(timeout)
        self._input_str = None
        thread = _Thread(target=self._take)
        thread.start()
        thread.join(timeout)

        if self._input_str is None and self._subproc is not None:
            self._subproc.terminate()

        return self._input_str


def load_json(from_file):
    """Loads the data from a JSON file to an object and returns the object.

    Args:
        from_file: the JSON file location

    Returns:
        result: the object
    """

    # Part of LYC-PythonUtils
    # Copyright 2022 Yucheng Liu. GNU GPL3 license.
    # GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt

    from_file = str(from_file)

    file = open(from_file, "r")
    obj = _jsonload(file)
    file.close()

    if isinstance(obj, dict):
        result = dict(obj)
    elif isinstance(obj, list):
        result = list(obj)
    elif isinstance(obj, str):
        result = str(obj)
    elif isinstance(obj, bool):
        result = bool(obj)
    elif isinstance(obj, int):
        result = int(obj)
    elif isinstance(obj, float):
        result = float(obj)
    elif isinstance(obj, _NoneType):
        result = None
    else:
        result = None
    # end if

    return result


def save_json(from_obj, to_file):
    """Saves the data from an object to a JSON file.

    Args:
        from_obj: the object
        to_file: the JSON file location
    """

    # Part of LYC-PythonUtils
    # Copyright 2022 Yucheng Liu. GNU GPL3 license.
    # GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt

    if isinstance(from_obj, dict):
        from_obj = dict(from_obj)
    elif isinstance(from_obj, list):
        from_obj = list(from_obj)
    elif isinstance(from_obj, str):
        from_obj = str(from_obj)
    elif isinstance(from_obj, bool):
        from_obj = bool(from_obj)
    elif isinstance(from_obj, int):
        from_obj = int(from_obj)
    elif isinstance(from_obj, float):
        from_obj = float(from_obj)
    elif isinstance(from_obj, _NoneType):
        from_obj = None
    else:
        from_obj = None
    # end if

    to_file = str(to_file)

    file = open(to_file, "w+")
    _jsondump(from_obj, file, indent=4)
    file.close()


def load_json_str(from_str):
    """Loads the data from a JSON string to an object and returns the object.

    Args:
        from_str: the JSON string

    Returns:
        result: the object
    """

    # Part of LYC-PythonUtils
    # Copyright 2022 Yucheng Liu. GNU GPL3 license.
    # GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt

    from_str = str(from_str)

    obj = _jsonloads(from_str)

    if isinstance(obj, dict):
        result = dict(obj)
    elif isinstance(obj, list):
        result = list(obj)
    elif isinstance(obj, str):
        result = str(obj)
    elif isinstance(obj, bool):
        result = bool(obj)
    elif isinstance(obj, int):
        result = int(obj)
    elif isinstance(obj, float):
        result = float(obj)
    elif isinstance(obj, _NoneType):
        result = None
    else:
        result = None
    # end if

    return result


def save_json_str(from_obj):
    """Saves the data from an object to a JSON string and return the string

    Args:
        from_obj: the object

    Returns:
        result: the JSON string
    """

    # Part of LYC-PythonUtils
    # Copyright 2022 Yucheng Liu. GNU GPL3 license.
    # GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt

    if isinstance(from_obj, dict):
        from_obj = dict(from_obj)
    elif isinstance(from_obj, list):
        from_obj = list(from_obj)
    elif isinstance(from_obj, str):
        from_obj = str(from_obj)
    elif isinstance(from_obj, bool):
        from_obj = bool(from_obj)
    elif isinstance(from_obj, int):
        from_obj = int(from_obj)
    elif isinstance(from_obj, float):
        from_obj = float(from_obj)
    elif isinstance(from_obj, _NoneType):
        from_obj = None
    else:
        from_obj = None
    # end if

    to_str = _jsondumps(from_obj, indent=4)

    result = to_str
    return result


def load_text(from_file):
    """Loads the data from a file to a string and returns the string.

    Args:
        from_file: the file location

    Returns:
        result: the text
    """

    # Part of LYC-PythonUtils
    # Copyright 2022 Yucheng Liu. GNU GPL3 license.
    # GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt

    from_file = str(from_file)

    file: _IO = open(from_file, "r")
    text = file.read()
    file.close()

    result = text
    result = str(result)
    return result


def save_text(from_str, to_file):
    """Saves a string to a file.

    Args:
        from_str: the string
        to_file: the file location
    """

    # Part of LYC-PythonUtils
    # Copyright 2022 Yucheng Liu. GNU GPL3 license.
    # GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt

    from_str = str(from_str)
    to_file = str(to_file)

    file: _IO = open(to_file, "w+")
    file.write(from_str)
    file.close()


def rand_bool():
    """Produce a random boolean value.

    This is like flipping a fair coin.

    Returns:
        result: the random boolean
    """

    # Part of LYC-PythonUtils
    # Copyright 2022 Yucheng Liu. GNU GPL3 license.
    # GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt

    result = bool(_randint(0, 1))
    return result


def logstr(logs, string=""):
    """Logs a string on the log file objects.

    Args:
        logs: the log file objects
        string: the string to log
    """

    # Part of LYC-PythonUtils
    # Copyright 2022 Yucheng Liu. GNU GPL3 license.
    # GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt

    logs = list(logs)
    string = str(string)

    for log in logs:
        log: _IO
        log.write(string)


def logln(logs, line=""):
    """Logs a line on the log file objects.

    Args:
        logs: the log file objects
        line: the line to log
    """

    # Part of LYC-PythonUtils
    # Copyright 2022 Yucheng Liu. GNU GPL3 license.
    # GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt

    logs = list(logs)
    line = str(line)

    line = line + "\n"

    for log in logs:
        log: _IO
        log.write(line)


def flushlogs(logs):
    """Flushes the logs.

    Args:
        logs: the log file objects
    """

    # Part of LYC-PythonUtils
    # Copyright 2022 Yucheng Liu. GNU GPL3 license.
    # GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt

    logs = list(logs)

    for log in logs:
        log: _IO
        log.flush()


def clamp_float(inval, bound1, bound2):
    """Clamps inval to the range bounded by bounds 1 and 2.

    Performs comparisons in floats.

    Args:
        inval: the input value
        bound1: bound 1
        bound2: bound 2

    Returns:
        result: the result
    """

    # Part of LYC-PythonUtils
    # Copyright 2022 Yucheng Liu. GNU GPL3 license.
    # GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt

    inval = float(inval)
    bound1 = float(bound1)
    bound2 = float(bound2)

    if bound1 < bound2:
        floor = bound1
        ceil = bound2
    else:  # elif bound1 >= bound2:
        floor = bound2
        ceil = bound1
    # end if

    result = inval

    if result < floor:
        result = floor

    if result > ceil:
        result = ceil

    return result


def clamp_int(inval, bound1, bound2):
    """Clamps inval to the range bounded by bounds 1 and 2.

    Performs comparisons in integers.

    Args:
        inval: the input value
        bound1: bound 1
        bound2: bound 2

    Returns:
        result: the result
    """

    # Part of LYC-PythonUtils
    # Copyright 2022 Yucheng Liu. GNU GPL3 license.
    # GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt

    inval = int(inval)
    bound1 = int(bound1)
    bound2 = int(bound2)

    if bound1 < bound2:
        floor = bound1
        ceil = bound2
    else:  # elif bound1 >= bound2:
        floor = bound2
        ceil = bound1
    # end if

    result = inval

    if result < floor:
        result = floor

    if result > ceil:
        result = ceil

    return result
