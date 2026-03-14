/*
 * directional_allocator.h
 * Alessio Burrello <alessio.burrello@unibo.it>
 * Luka Macan <luka.macan@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 */


/* Directional allocator
 *
 *  Allocates memory depending on the direction argument:
 *    - direction == 1: allocates from the beginning of the memory
 *    - direction == 0: allocates from the ending of the memory
 */

#ifndef _DIRECTIONAL_ALLOCATOR_H
#define _DIRECTIONAL_ALLOCATOR_H

#include <stdint.h>
#include <stddef.h>

#define NULL (void *)0

typedef struct {
    uintptr_t base_addr;
    uintptr_t limit_addr;
    uintptr_t begin_addr;
    uintptr_t end_addr;
    uintptr_t last_before_begin_addr;
    uintptr_t last_before_end_addr;
    uintptr_t last_after_begin_addr;
    uintptr_t last_after_end_addr;
    uintptr_t last_return_addr;
    size_t last_request_size;
    uint32_t last_alignment;
    int32_t last_direction;
    int32_t last_operation;
} directional_allocator_debug_state_t;

static uint8_t *directional_mem_base = NULL;
static uint8_t *directional_mem_limit = NULL;
static uint8_t *directional_mem_begin = NULL;
static uint8_t *directional_mem_end = NULL;
static directional_allocator_debug_state_t directional_allocator_debug = {0};

static void directional_allocator_get_debug_state(directional_allocator_debug_state_t *out) {
    if (out == NULL) {
        return;
    }
    *out = directional_allocator_debug;
}


static void directional_allocator_init(void *begin, int size) {
    directional_mem_base = (uint8_t *) begin;
    directional_mem_limit = directional_mem_base + size;
    directional_mem_begin = directional_mem_base;
    directional_mem_end = directional_mem_limit;
    directional_allocator_debug.base_addr = (uintptr_t) directional_mem_base;
    directional_allocator_debug.limit_addr = (uintptr_t) directional_mem_limit;
    directional_allocator_debug.begin_addr = (uintptr_t) directional_mem_begin;
    directional_allocator_debug.end_addr = (uintptr_t) directional_mem_end;
    directional_allocator_debug.last_before_begin_addr = (uintptr_t) directional_mem_begin;
    directional_allocator_debug.last_before_end_addr = (uintptr_t) directional_mem_end;
    directional_allocator_debug.last_after_begin_addr = (uintptr_t) directional_mem_begin;
    directional_allocator_debug.last_after_end_addr = (uintptr_t) directional_mem_end;
    directional_allocator_debug.last_return_addr = 0;
    directional_allocator_debug.last_request_size = 0;
    directional_allocator_debug.last_alignment = 1;
    directional_allocator_debug.last_direction = 0;
    directional_allocator_debug.last_operation = 0;
#if DBG_DIRALLOC
    printf("Directional allocator init:\n   Set directional_mem_begin to 0x%X\n   Set directional_mem_end to 0x%X", directional_mem_begin, directional_mem_end);
#endif
}

static void *dmalloc(int size, int direction) {
    void *retval = NULL;
    directional_allocator_debug.last_before_begin_addr = (uintptr_t) directional_mem_begin;
    directional_allocator_debug.last_before_end_addr = (uintptr_t) directional_mem_end;
    directional_allocator_debug.last_request_size = (size_t) size;
    directional_allocator_debug.last_alignment = 1;
    directional_allocator_debug.last_direction = direction;
    directional_allocator_debug.last_operation = 1;
    if (directional_mem_begin + size < directional_mem_end) {
        if (direction == 1) {
            retval = directional_mem_begin;
            directional_mem_begin += size;
        } else {
            directional_mem_end -= size;
            retval = directional_mem_end;
        }
#if DBG_DIRALLOC
        printf("Direcional allocator:\n   Allocated %d bytes in direction %d\n   Begin now at 0x%X\n   End now at 0x%X\n", size, direction, directional_mem_begin, directional_mem_end);
#endif
    }
    directional_allocator_debug.begin_addr = (uintptr_t) directional_mem_begin;
    directional_allocator_debug.end_addr = (uintptr_t) directional_mem_end;
    directional_allocator_debug.last_after_begin_addr = (uintptr_t) directional_mem_begin;
    directional_allocator_debug.last_after_end_addr = (uintptr_t) directional_mem_end;
    directional_allocator_debug.last_return_addr = (uintptr_t) retval;
    return retval;
}

static void dfree(int size, int direction) {
    directional_allocator_debug.last_before_begin_addr = (uintptr_t) directional_mem_begin;
    directional_allocator_debug.last_before_end_addr = (uintptr_t) directional_mem_end;
    directional_allocator_debug.last_request_size = (size_t) size;
    directional_allocator_debug.last_alignment = 1;
    directional_allocator_debug.last_direction = direction;
    directional_allocator_debug.last_operation = 2;
    if (direction == 1)
        directional_mem_begin -= size;
    else
        directional_mem_end += size;
    directional_allocator_debug.begin_addr = (uintptr_t) directional_mem_begin;
    directional_allocator_debug.end_addr = (uintptr_t) directional_mem_end;
    directional_allocator_debug.last_after_begin_addr = (uintptr_t) directional_mem_begin;
    directional_allocator_debug.last_after_end_addr = (uintptr_t) directional_mem_end;
    directional_allocator_debug.last_return_addr = 0;
#if DBG_DIRALLOC
    printf("Directional allocator:\n   Freed %d bytes in direction %d\n   Begin now at 0x%X\n   End now at 0x%X\n", size, direction, directional_mem_begin, directional_mem_end);
#endif
}

#endif
