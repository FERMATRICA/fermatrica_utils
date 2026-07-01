import sys
import pytest

from fermatrica_utils.objects import StableClass, get_size


class TestStableClass:

    def test_set_predefined_attr_works(self):
        class MyClass(StableClass):
            def __init__(self):
                self.x = 0
                self._init_finish()

        obj = MyClass()
        obj.x = 99
        assert obj.x == 99

    def test_set_new_attr_after_init_raises(self):
        class MyClass(StableClass):
            def __init__(self):
                self.x = 0
                self._init_finish()

        obj = MyClass()
        with pytest.raises(AttributeError):
            obj.new_attr = 'forbidden'

    def test_set_attr_before_init_finish_allowed(self):
        class MyClass(StableClass):
            def __init__(self):
                self.a = 1
                self.b = 2
                self._init_finish()

        obj = MyClass()
        assert obj.a == 1
        assert obj.b == 2

    def test_init_finish_not_called_allows_new_attrs(self):
        class MyClass(StableClass):
            def __init__(self):
                self.x = 0

        obj = MyClass()
        obj.new_attr = 'allowed'
        assert obj.new_attr == 'allowed'

    def test_two_instances_are_independent(self):
        class MyClass(StableClass):
            def __init__(self, val):
                self.x = val
                self._init_finish()

        a = MyClass(1)
        b = MyClass(2)
        assert a.x == 1
        assert b.x == 2
        with pytest.raises(AttributeError):
            a.y = 'bad'

    def test_error_message_contains_class_name(self):
        class SpecificClass(StableClass):
            def __init__(self):
                self.x = 0
                self._init_finish()

        obj = SpecificClass()
        with pytest.raises(AttributeError, match='SpecificClass'):
            obj.forbidden = 'x'


class TestGetSize:

    def test_primitive_returns_positive_int(self):
        size = get_size(42)
        assert isinstance(size, int)
        assert size > 0

    def test_list_with_items_larger_than_empty(self):
        size_empty = get_size([])
        size_with_items = get_size([1, 2, 3, 4, 5])
        assert size_with_items > size_empty

    def test_nested_dict_returns_positive(self):
        size = get_size({'a': {'b': [1, 2, 3]}, 'c': 'hello'})
        assert size > 0

    def test_string_size_grows_with_length(self):
        size_short = get_size('hi')
        size_long = get_size('hi' * 1000)
        assert size_long > size_short

    def test_none_returns_positive_int(self):
        size = get_size(None)
        assert isinstance(size, int)
        assert size > 0

    def test_circular_reference_does_not_infinite_loop(self):
        a = []
        a.append(a)
        size = get_size(a)
        assert isinstance(size, int)
        assert size > 0

    def test_empty_dict_smaller_than_populated_dict(self):
        size_empty = get_size({})
        size_full = get_size({'a': 1, 'b': 2, 'c': [1, 2, 3]})
        assert size_full > size_empty

    def test_object_with_slots(self):
        class Slotted:
            __slots__ = ['x', 'y']

            def __init__(self):
                self.x = 42
                self.y = 'hello'

        size = get_size(Slotted())
        assert isinstance(size, int)
        assert size > 0
