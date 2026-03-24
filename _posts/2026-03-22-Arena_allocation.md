# Arena Allocation in C

Arena allocation (also called **region allocation** or **bump allocation**) is a memory management strategy where a large block of memory is reserved upfront and smaller allocations are carved out sequentially from it.

Its biggest advantage is the you don't need to allocate memory multiple times (allocating memory is expensive and handled
by asking the OS for space in memory pages), nor free multiple times, relieving the programmer from the concern of freeing
each variable individually. Everything is cleared/freed at once by resetting/destroying the arena.


## 1. Core Idea

An arena allocator works as follows:

- Reserve one large contiguous block of memory (`malloc`, `mmap`, etc.).
- Maintain a pointer (or offset) to the current free position.
- On each allocation:
  - Return memory at the current pointer.
  - Advance the pointer by the requested size (plus alignment padding).
  - Optionally: if the pointer exceeds the capacity, create and point to the next block.
- Deallocation: memory is released in bulk by resetting the current pointer or destroying the arena.

Because allocation simply advances a pointer, this is often called a **bump allocator**.
Allocation cost is obviously O(1).


## 2. The arena data structure

Conceptually, the minimal structure the arena looks like:

```c
typedef struct {
    char* base;
    size_t capacity;
    size_t offset;
} Arena;
```

- `void* base` -- start of the memory region
- `size_t capacity` -- total size of the region
- `size_t offset` -- number of bytes currently used

This block holds all allocations managed by the arena.
This determines where the next allocation will occur.
This offset is with respect to the arena base - it is not an absolute address.

Therefore there are no per-allocation metadata, no block splitting or merging and
no linksed list/tree structures. Allocation is purely sequential.


## 3. Schematic view

You can imagine the arena memory as one large buffer. Initially:

```
[--------------------------------------------------------]
^                                                        ^
base                                                     base + capacity
^
offset = 0
```

After allocating 16 bytes:

```
[xxxxxxxx-----------------------------------------------]
^       ^
base    offset = 16
```

After allocating another 32 bytes:

```
[xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx------------------------]
^                               ^
base                            offset = 48
```

Each allocation: returns a void pointer at `base + offset` and advances `offset` by the allocation size.

## 4. Alignment Handling

Allocations must respect alignment requirements. Arena allocators allow the allocation of arbitrary numbers of bytes
(as long as the new size doesn't exceed the capacity).
However, CPUs fetch and write data from addresses that are  multiples 
of a fixed alignment size (e.g. 16 bytes)
to load data faster or cache it more efficiently.

Before returning memory:

1. The current pointer is rounded up to the required alignment.
2. The offset is advanced accordingly.
3. Return the referenced memory as a void pointer (so it can be cast to the desired data by the caller).

Conceptually:

```c
aligned = align_up(current, alignment);
offset = (aligned - base) + size;
```

Alignment may introduce small internal padding gaps.
One subtle but important detail is how alignment is implemented in binary operations.

Remember that the offset is measured w.r.t to the base. For example 
if the areana was allocated from addresses 800 to 1200, then offset of 267 
corresponds to current address of `800 + 267 = 1067`.

```
       +-+-+-+-+-+-+-+-+-+-+-+
1067 = |1|0|0|0|0|1|0|1|0|1|1|
       +-+-+-+-+-+-+-+-+-+-+-+
```

Let's say after an allocation we ended with reserved 
cells up to position `p = 1067` in the areana block.
Assume alignment every 8 bytes. Then we must mark the 
entire "row" of the pointer as reserved and advance it
(round it up) to the next row.
       
```
           (unaligned)    X = reserved cells
             offset 
                |
old base        |
   ^            |
   |     1064   v         1071     1064             1071
   +-------+-+-+-+-+-+-+-+-+         +-+-+-+-+-+-+-+-+
           |X|X|X| | | | | |         |X|X|X|X|X|X|X|X|
           +-+-+-+-+-+-+-+-+         +-+-+-+-+-+-+-+-+
           | | | | | | | | |         | | | | | | | | |
           +-+-+-+-+-+-+-+-+         +-+-+-+-+-+-+-+-+
         1072             1079   1072^              1079
                                     |
                                     +----- aligned offset
                                     |
                                     +----- new base
```      

The question is, given a current offset, how can we  find its aligned offset and 
do this with binary operations?

The steps are:

1. Find current address: 
```c
uintptr_t current = (uintptr_t)(arena->base + arena->offset);
current = 800 + 276 = 1067
```

2. Add `alignment - 1` -> we end up in the next row, e.g.
   `1067 -> 1067 + 7 = 1074`.
3. Round down to the beginning of this row. To do this, we need to get
   rid of its 3 LSBs, i.e. AND it with `~7 = ~(alignment - 1)`.

```
    +-+-+-+-+-+-+-+-+-+-+-+
    |1|0|0|0|0|1|0|1|0|1|1| = offset          = 1076
    +-+-+-+-+-+-+-+-+-+-+-+
AND |1|1|1|1|1|1|1|1|0|0|0| = alignment - 1   = ~7
    +-+-+-+-+-+-+-+-+-+-+-+
    |1|0|0|0|0|1|0|1|0|0|0| = aligned address = 1064
    +-+-+-+-+-+-+-+-+-+-+-+
```

In C:
```
uintptr_t aligned = (current + alignment - 1) & ~(alignment - 1);
```

4. We ended up in the beginning of the current row, which should be 
   reserved. Therefore shift the offset to the beginning of the next row:

```
size_t new_offset = (aligned - (uintptr_t)arena->base) + size;
```


## 5. Growing Arenas (Multi-Block Arenas)

A fixed-size arena fails when the offset reaches the capacity. To address this, we use growing arenas.
The blueprint is as follows:

```c
typedef struct ArenaBlock {
    char* memory;
    size_t capacity;
    size_t offset;
    struct ArenaBlock* next;
} ArenaBlock;
```

When a block fills:

- We compure the capacity of the next block, which is typically twice the capacity of the current.
- Then we create a next block, which is referenced by the `next` node of the current one.
- Then we check if there's enough capacity in the new block and either write to this block (and advance the current pointer to `(aligned - base) + size`) or repeat creating a new one.

The head always points at the first block. In multi-block areanas, we iterate for space linearly. To avoid this linearty, we can optionally introduce a `tail` pointer (not implemented here for simplicity).

## 6. Multi-block arena implementation

```c
#include <stdio.h>
#include <stdlib.h> // malloc, free
#include <stdint.h> // SIZE_MAX, uintptr_t
#include <stddef.h> // max_align_t, _Alignof, size_t

#define CHECK_ALLOC(ptr, arena_ptr) \
  do { if (!(ptr)) { arena_destroy(arena_ptr); return 1; } } while (0)

typedef struct {
  const char* name;
  unsigned id;
} Data;

typedef struct Arena {
  char* base;
  size_t capacity;
  size_t offset;
  struct Arena* next;
} Arena;

Arena* arena_create(size_t size) { 
  Arena* ret = (Arena*) malloc(sizeof(Arena));
  if (!ret) {
    return NULL;
  }
  ret->base = (char*) malloc(size);
  if (!ret->base) {
    free(ret);
    return NULL;
  }
  ret->capacity = size;
  ret->offset = 0;
  ret->next = NULL;
  return ret;
}

void arena_reset(Arena* arena) {
  while (arena) {
    arena->offset = 0;
    arena = arena->next;
  }
}

void arena_destroy(Arena* arena) {
    while (arena) {
        Arena* next = arena->next;
        free(arena->base);
        free(arena);
        arena = next;
    }
}

void* arena_alloc(Arena* arena, size_t size, size_t alignment) {
    if (!arena || alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return NULL;
    }

    // helper pointer to add new nodes
    Arena* current_arena = arena;
    while (current_arena) {
        // as an absolute address
        uintptr_t current_ptr = (uintptr_t)(current_arena->base + current_arena->offset);
        // round up to the next multiple of alignment
        uintptr_t aligned = (current_ptr + alignment - 1) & ~(alignment - 1);
        // relative to base
        size_t new_offset = (aligned - (uintptr_t)current_arena->base) + size;
        if (new_offset <= current_arena->capacity) {
          current_arena->offset = new_offset;
          return (void*)aligned;
        }

        // we have exceeded the capacity -
        //calculate next capacity before creating next node
        if (!current_arena->next) {
            size_t required = size + alignment - 1;
            // keep doubling the capacity till we exceed the required size
            size_t next_capacity = current_arena->capacity * 2;
            while (next_capacity < required) {
                // SIZE_MAX is the largest allocation we can make
                if (next_capacity > SIZE_MAX / 2) {
                    next_capacity = required;
                    break;
                }
                next_capacity *= 2;
            }
            current_arena->next = arena_create(next_capacity);
            if (!current_arena->next)
              return NULL;
        }
        current_arena = current_arena->next;
    }
    return NULL;
}


int main(int argc, char *argv[])
{
  Arena* arena = arena_create(64);
  if (!arena) {
    fprintf(stderr, "Failed to create arena\n");
    return 1;
  }

  Data* data1 = (Data*)arena_alloc(arena, sizeof(Data), _Alignof(Data));
  CHECK_ALLOC(data1, arena);
  Data* data2 = (Data*)arena_alloc(arena, sizeof(Data), _Alignof(Data));
  CHECK_ALLOC(data2, arena);
  *data1 = (Data) {.name = "Alice", .id = 41};
  *data2 = (Data) {.name = "Bob",   .id = 42}; 

  // generic block of data - max_align_t aligns safely all types
  void* big = arena_alloc(arena, 80, _Alignof(max_align_t));
  CHECK_ALLOC(big, arena);

  Data* data3 = (Data*)arena_alloc(arena, sizeof(Data), _Alignof(Data));

  CHECK_ALLOC(data3, arena);
  data3->name = "Snoop Dog";
  data3->id = 420;

  Data* data4 = (Data*)arena_alloc(arena, sizeof(Data), _Alignof(Data));
  data4->name = "Still in base block";
  data4->id = 43;

  Data* data5 = (Data*)arena_alloc(arena, sizeof(Data), _Alignof(Data));
  data5->name = "Now in next block";
  data5->id = 44;

  // overwrite all previous data
  arena_reset(arena); 
  Data* data6 = (Data*)arena_alloc(arena, sizeof(Data), _Alignof(Data));
  data6->name = "Now in base block";
  data6->id = 45;

  arena_destroy(arena);
  return 0;
}
```


## 7. Inspecting the arena

By compiling with the `-g` flag, setting breakpoints at `arena_reset(arena)` and `arena_destroy(arena)`, we can inspect
how the blocks have been filled and what block each variable gets assigned to. The capacity of the first block is 64 and the size of each `Data` structure is 16.
After allocating the first two `Data` structure we occpupy 32/64 bytes, so by allocating a `void*` block of data of size 80, we would fill the head block. the `void*` gets allocated to arena `arena->next` and so on. After resetting the arena, we 
return to offset 0 of the head block. This is confirmed by inspecting it with `gdb`:

```
(gdb) p *(Data*)((char*)arena->base + (arena->offset - 3*sizeof(Data)))
$1 = {name = 0x555555556022 "Bob", id = 42}
(gdb) p *(Data*)((char*)arena->base + (arena->offset - 4*sizeof(Data)))
$2 = {name = 0x55555555601c "Alice", id = 41}
(gdb) p *(Data*)((char*)arena->base + (arena->offset - 1*sizeof(Data)))
$3 = {name = 0x555555556030 "Still in base block", id = 43}
(gdb) p *(Data*)((char*)arena->base + (arena->offset - 2*sizeof(Data)))
$4 = {name = 0x555555556026 "Snoop Dog", id = 420}
(gdb) p *(Data*)((char*)arena->next->base + arena->next->offset - sizeof(Data))
$13 = {name = 0x555555556044 "Now in next block", id = 44}
(gdb) p arena->next->offset - (80 + sizeof(Data))
$16 = 0
(gdb) p *(Data*)((char*)arena->base + (arena->offset - 1*sizeof(Data)))
$17 = {name = 0x555555556056 "Now in first block", id = 45}
```

In summary:

```
sizeof data = 16
capacity of base block = 64
-----------------------------
after data1: 16/64
after data2: 32/64
after big: 80 > 64 - 32 => we allocate it in the next block, of capacity 128
next block: 80/128
after data3: 48/64
after data4: 64/64
after data5: base block full => search in next block -> 96/128
after reset: base block 0/64
after data6: 16/64
```
