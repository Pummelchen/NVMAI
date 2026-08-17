#ifndef NVMAI_EXPERT_IO_H
#define NVMAI_EXPERT_IO_H

#include <stddef.h>
#include <stdint.h>

/// Parallel reader for routed-expert blocks.
///
/// v3.x fetched misses one at a time on the calling thread and reached ~0.62 GB/s.
/// Measured on the same machine and the same file, expert-sized (1.688 MiB) reads
/// sustain ~2.0 GB/s on one thread and ~3.2 GB/s on four, and random offsets are
/// as fast as sequential ones -- NVMe does not care about locality at this
/// granularity, so nothing needs reordering. The gap was concurrency, not the
/// device.
///
/// Reads go through `pread` on per-thread descriptors: no shared file offset, no
/// lseek, and no mmap, so a fetch cannot fault the GPU's address space or leave
/// page-cache pressure behind for the rest of the machine to pay. That matters
/// here because the point of streaming is to keep RAM free.
typedef struct nvmai_expert_reader nvmai_expert_reader;

/// Opens `path` with `threads` independent descriptors.
///
/// Returns NULL if the file cannot be opened or the arguments are invalid;
/// `out_errno` receives the failure cause when non-NULL. `threads` is clamped to
/// [1, 16] -- four saturates this device and more does not help.
/// `bypass_cache` sets `F_NOCACHE`, keeping expert reads out of the unified
/// buffer cache.
///
/// This is a policy choice, not an optimisation. With the page cache in play,
/// repeated reads of a hot expert are served from RAM and are very fast -- but the
/// cache grows without bound and consumes exactly the memory this project exists to
/// leave free. With it bypassed, the slot cache is the *only* cache, so the
/// declared RAM budget is the true footprint and throughput is the device's.
///
/// Default to bypassing for a predictable footprint; allow the page cache when the
/// caller would rather have the free speed.
nvmai_expert_reader *nvmai_expert_reader_create(const char *path,
                                               size_t expert_stride,
                                               int threads,
                                               int bypass_cache,
                                               int *out_errno);

void nvmai_expert_reader_destroy(nvmai_expert_reader *reader);

/// Reads `count` experts, `expert_ids[i]` into `destinations[i]`, and blocks
/// until every one has completed.
///
/// Each destination must hold `expert_stride` bytes. Returns 0 on success, or the
/// first `errno` observed by any worker; on failure the contents of the
/// destinations are undefined and the caller must not use them.
int nvmai_expert_reader_fetch(nvmai_expert_reader *reader,
                              const uint32_t *expert_ids,
                              void *const *destinations,
                              size_t count);

/// As `nvmai_expert_reader_fetch`, but the caller supplies absolute byte offsets.
///
/// Callers that lay experts out as `index * stride` can use the id form; the
/// streamer cannot, because its regions carry a per-layer base and a container
/// offset, so an id would silently address the wrong layer.
int nvmai_expert_reader_fetch_offsets(nvmai_expert_reader *reader,
                                     const uint64_t *offsets,
                                     void *const *destinations,
                                     size_t count);

/// Threads actually in use, after clamping.
int nvmai_expert_reader_threads(const nvmai_expert_reader *reader);

#endif /* NVMAI_EXPERT_IO_H */
