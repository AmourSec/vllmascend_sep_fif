// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>

extern "C" {
int a5_open(int device, uint64_t bytes, void** state);
int a5_close(void* state);
int a5_attach(void* state, uint64_t hva, uint32_t flags);
int a5_bind_existing(void* state, uint64_t hva, uint64_t dva);
int a5_pinned(void* state);
int a5_bare_pid(void* state, int32_t* pid);
int a5_peer(void* state, int peer);
int a5_vmm_create(void* state, int numa, int huge, const int32_t* pids, size_t count, uint64_t* shared,
                  uint64_t* physical_bytes);
int a5_vmm_import(void* state, uint64_t shared, uint64_t physical_bytes);
uint64_t a5_hva(void* state);
uint64_t a5_dva(void* state);
uint64_t a5_hbm(void* state);
int a5_read(void* state, void* output, int path);
int a5_write(void* state, const void* input, uint64_t offset, uint64_t bytes, int path);
int a5_upload(void* state, const void* input);
}
