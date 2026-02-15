from __future__ import annotations
from typing import Final, Generic, TypeVar, Optional, Sequence
from numbers import Number
from collections import deque
import math
import numpy as np

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
        self._size = 0
        #self._items: list[Optional[T]] = [None] * capacity
        self._items = np.zeros(capacity, dtype=np.int16)
        self._OVERWRITING = OVERWRITING
        self._RESIZE = RESIZE
        self._DEBUG = DEBUG

        self._head = 0 #Write Pointer
        self._tail = 0 #Read Pointer

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

    def bulk_enqueue(self, items: Sequence[T]) -> None:
        """
        Add multiple elements to the queue at once.

        Args:
            items (Sequence[T]): Elements to add.

        Returns:
            int: Number of elements actually enqueued.
        """
        if not items: return

        input_size: int = len(items)
        space_in_buffer  : int = self._capacity - self._size

        if self._RESIZE and input_size > space_in_buffer:
            self.resize(max(self._capacity * 2, self._size + 1))

        if not self._OVERWRITING and input_size > space_in_buffer:
            input_size = space_in_buffer
            items = items[:space_in_buffer]

        if input_size <= 0: return

        #Overwriting buffer for overflow
        overflow = max(0, self._size + input_size - self._capacity)
        if overflow:
            self._tail = self._move_pointer(self._tail, overflow)
            self._size -= overflow 
        first_part = min(input_size, self._capacity - self._head)
        second_part = input_size - first_part

        self._items[self._head : self._head + first_part] = items[:first_part]

        if second_part:
            self._items[:second_part] = items[first_part:]

        self._head = self._move_pointer(self._head, input_size)
        self._size += input_size 

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
            #out += self._items[0:remaining]
            out = np.concatenate((
                self._items[tail : tail + first_part],
                self._items[0:remaining]
            ))

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

    def print_circle(self, radius: int = 10) -> None:
        if self.is_empty() and not self._DEBUG:
            print("\n[ Buffer is empty ]\n")
            return

        aspect_ratio = 2.0 
        width = int(radius * aspect_ratio * 2) + 15
        height = (radius * 2) + 5
        center_x, center_y = width // 2, height // 2
        canvas = [[" " for _ in range(width)] for _ in range(height)]

        for i in range(self._capacity):
            angle = (2 * math.pi * i / self._capacity) - (math.pi / 2)
            x = int(center_x + radius * math.cos(angle) * aspect_ratio)
            y = int(center_y + radius * math.sin(angle))

            is_head = (i == self._head % self._capacity)
            is_tail = (i == self._tail % self._capacity)
            val = self._items[i]
            display_val = f"({val})" if val is not None else "[ ]"
            
            ptr = ""
            if is_head and is_tail: ptr = "H+T"
            elif is_head: ptr = "H"
            elif is_tail: ptr = "T"
            
            label = f"{ptr:^3} {display_val}" if ptr else display_val
            start_x = x - (len(label) // 2)
            for char_idx, char in enumerate(label):
                if 0 <= y < height and 0 <= (start_x + char_idx) < width:
                    canvas[y][start_x + char_idx] = char

        title = f" CAPACITY: {self._capacity} "
        canvas[center_y][center_x - len(title)//2 : center_x + (len(title)+1)//2] = list(title)

        border = "=" * width
        print(f"\n{border}")
        for row in canvas: print("".join(row))
        print(border)
        print(f"  Legend: H=Head | T=Tail | Size: {self._size}")
        print(f"{border}\n")

if __name__ == "__main__":
    d = CircularBuffer(5, None, True, False, True)
    d.bulk_enqueue([1,2,3,4,5,6])
    d.print_circle()