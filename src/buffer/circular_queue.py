from __future__ import annotations
from typing import Final, Generic, TypeVar, Optional, Sequence
from numbers import Number
from collections import deque
from threading import Lock
import math

T = TypeVar("T", bound=Number)

class CircularBuffer(Generic[T]):
    """
    A thread-safe circular queue (ring buffer) implementation with optional 
    overwriting and resizing.

    Attributes:
        _items (list[Optional[T]]): Internal storage for the queue elements.
        _capacity (int): Maximum number of elements the queue can hold.
        _head (int): Index of the next position to enqueue.
        _tail (int): Index of the next position to dequeue.
        _size (int): Current number of elements in the queue.
        _OVERWRITING (bool): Whether to overwrite oldest elements when full.
        _RESIZE (bool): Whether to automatically resize the queue when full.
        _DEBUG (bool): Whether to raise exceptions on invalid operations.
        _POW_2 (bool): Whether the capacity is a power of 2 (for faster modulo).
        lock (Lock): Threading lock for concurrent access.
    """

    __slots__ = {"_items", "_OVERWRITING", "_head", "_tail", "_size", "_capacity", "_POW_2", "_RESIZE", "_DEBUG", "lock"}

    def __init__(
        self, 
        capacity: Optional[int], 
        items: Optional[Sequence[T]] = None, 
        OVERWRITING: Optional[bool] = False, 
        RESIZE: Optional[bool] = False, 
        DEBUG: Optional[bool] = False
    ) -> None:
        """
        Initialize the CircularQueue.

        Args:
            capacity (Optional[int]): Maximum capacity of the queue.
            items (Optional[Sequence[T]]): Initial elements to populate the queue.
            OVERWRITING (Optional[bool]): Overwrite old elements if full.
            RESIZE (Optional[bool]): Automatically resize if full.
            DEBUG (Optional[bool]): Enable debug checks.

        Raises:
            ValueError: If both capacity and items are None or invalid values.
        """
        if capacity is None and items is None and DEBUG: 
            raise ValueError("Capacity or items must be passed in")
        
        if capacity is None: 
            capacity = len(items)

        if capacity <= 0 and DEBUG:
            raise ValueError("Capacity must be positive")

        self._capacity = capacity
        self._items: list[Optional[T]] = [None] * capacity

        self._OVERWRITING = OVERWRITING
        self._RESIZE = RESIZE
        self._DEBUG = DEBUG

        self._head = 0
        self._tail = 0
        self._size = 0

        self._POW_2 = self._is_power_of_2()

        if items:
            if len(items) > capacity and DEBUG:
                raise ValueError("Items exceed capacity")
            
            for value in items:
                self.enqueue(value)

    def _is_power_of_2(self) -> bool:
        """Check if the capacity is a power of 2."""
        return self._capacity > 0 and (self._capacity & (self._capacity - 1)) == 0
    
    def _move_pointer(self, pointer: int, step: int = 1) -> int:
        """
        Move a pointer forward by a given step, wrapping around the queue.

        Args:
            pointer (int): Current index.
            step (int): Steps to move forward.

        Returns:
            int: New index after moving.
        """
        if self._POW_2:
            return (pointer + step) & (self._capacity - 1)
        else:
            return (pointer + step) % self._capacity

    def peek(self) -> T:
        """
        Get the element at the tail without dequeuing.

        Returns:
            T: Element at the tail.

        Raises:
            ValueError: If the queue is empty and DEBUG is True.
        """
        if self.is_empty() and self._DEBUG:
            raise ValueError("Circular Buffer is empty")
        return self._items[self._tail]
    
    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return self._size == 0

    def is_full(self) -> bool:
        """Check if the queue is full."""
        return self._size == self._capacity

    def enqueue(self, value: T) -> None:
        """
        Add an element to the queue.

        Args:
            value (T): Element to add.

        Raises:
            ValueError: If queue is full and neither OVERWRITING nor RESIZE is enabled (DEBUG=True).
        """
        if self.is_full():
            if self._OVERWRITING:
                self._tail = self._move_pointer(self._tail)
            elif self._RESIZE:
                self.resize()
            else:
                if self._DEBUG:
                    raise ValueError("Queue is full")
        else:
            self._size += 1

        self._items[self._head] = value
        self._head = self._move_pointer(self._head)

    def bulk_enqueue(self, items: Sequence[T]) -> int:
        """
        Add multiple elements to the queue at once.

        Args:
            items (Sequence[T]): Elements to add.

        Returns:
            int: Number of elements actually enqueued.
        """
        input_size: int = len(items)

        if input_size == 0:
            return 0

        space: int = self._capacity - self._size

        if input_size > space:
            if self._OVERWRITING:
                overflow: int = input_size - space
                for _ in range(overflow):
                    self._tail = self._move_pointer(self._tail)
            elif self._RESIZE:
                self.resize(self._capacity + (input_size - space))
                space = self._capacity - self._size
            else:
                input_size = space

        count: int = min(input_size, space)
        for i in range(count):
            self._items[self._head] = items[i]
            self._head = self._move_pointer(self._head)

        self._size += count
        return count

    def dequeue(self) -> T:
        """
        Remove and return the element at the tail.

        Returns:
            T: The dequeued element.

        Raises:
            ValueError: If queue is empty and DEBUG=True.
        """
        if self.is_empty() and self._DEBUG:
            raise ValueError("Circular Buffer is empty")
        
        value: T = self._items[self._tail]
        self._tail = self._move_pointer(self._tail)
        self._size -= 1

        return value
    
    def bulk_dequeue(self, amount: int) -> list[T]:
        """
        Remove and return multiple elements from the tail.

        Args:
            amount (int): Number of elements to dequeue.

        Returns:
            list[T]: List of dequeued elements.
        """
        count: int = min(amount, self._size)

        if count == 0: return []

        tail = self._tail
        capacity = self._capacity
        
        first_part = min(count, capacity - tail) #Check if tail is close to end, and slice from tail to end
        out = self._items[tail : tail + first_part]

        remaining = count - first_part
        if remaining > 0:
            out += self._items[0:remaining]

        self._tail = self._move_pointer(self._tail, count)
        self._size -= count

        return out
        
    def resize(self, new_capacity: Optional[int] = None) -> None:
        """
        Resize the queue to a new capacity.

        Args:
            new_capacity (Optional[int]): New capacity. If None, doubles the current capacity.
        """
        if not self._RESIZE:
            return

        if new_capacity is None:
             new_capacity = max(self._capacity * 2, self._size + 1)

        if not isinstance(new_capacity, int):
            raise TypeError(f"Capacity must be int, got {type(new_capacity)}: {new_capacity}")
        print(new_capacity)
        new_items: list[Optional[T]] = [None] * new_capacity

        for i in range(self._size):
            new_items[i] = self._items[(self._tail + i) % self._capacity]
        
        self._items = new_items
        self._capacity = new_capacity
        self._head = self._size
        self._tail = 0
        self._POW_2 = self._is_power_of_2()

    def print(self) -> None:
        """Print the internal array of the queue (debugging)."""
        print(self._items)

    def print_circle(self, _radius: Optional[int] = None) -> None:
        """
        Print a simple text-based circle representation of the queue.

        Args:
            _radius (Optional[int]): Radius of the circle. Defaults to max(3, capacity // 2).
        """
        if self.is_empty():
            print("Circular Queue is empty")
            return

        n = self._capacity
        radius = _radius if _radius else max(3, n // 2)  
        center = radius
        canvas_size = radius * 2 + 1

        canvas = [[' ' for _ in range(canvas_size)] for _ in range(canvas_size)]

        y_scale = 0.5 

        for i in range(n):
            angle = 2 * math.pi * i / n
            x = round(center + radius * math.cos(angle))
            y = round(center + radius * math.sin(angle) * y_scale)
            idx = (self._tail + i) % n
            item = self._items[idx]
            if 0 <= y < canvas_size and 0 <= x < canvas_size:
                canvas[y][x] = str(item) if item is not None else '.'

        head_x = round(center + radius * math.cos(2 * math.pi * (self._head - 1) / n))
        head_y = round(center + radius * math.sin(2 * math.pi * (self._head - 1) / n) * y_scale)
        tail_x = round(center + radius * math.cos(2 * math.pi * self._tail / n))
        tail_y = round(center + radius * math.sin(2 * math.pi * self._tail / n) * y_scale)

        if 0 <= head_y < canvas_size and 0 <= head_x < canvas_size and \
           0 <= tail_y < canvas_size and 0 <= tail_x < canvas_size:
            if canvas[head_y][head_x] == canvas[tail_y][tail_x]:
                canvas[head_y][head_x] = 'B' 
            else:
                canvas[head_y][head_x] = 'H'
                canvas[tail_y][tail_x] = 'T'

        for row in canvas:
            print(''.join(row))
