// Parallel routed-expert reader. See include/nvmai_expert_io.h for why this
// exists: the device sustains ~3.2 GB/s on four concurrent expert-sized reads and
// v3.x was reaching ~0.62 GB/s by fetching one at a time on the calling thread.

#include "include/nvmai_expert_io.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define NVMAI_IO_MAX_THREADS 16

struct nvmai_expert_reader {
    int fds[NVMAI_IO_MAX_THREADS];
    int threads;
    size_t expert_stride;

    pthread_t workers[NVMAI_IO_MAX_THREADS];
    pthread_mutex_t lock;
    pthread_cond_t work_ready;   // a batch was published, or shutdown
    pthread_cond_t work_done;    // outstanding hit zero

    // Current batch. Valid only while outstanding > 0.
    const uint32_t *expert_ids;
    void *const *destinations;
    size_t count;
    size_t next_index;           // claimed by workers
    size_t outstanding;          // published minus completed
    int first_errno;
    int shutting_down;
    uint64_t generation;         // so a worker cannot re-run a finished batch
};

/// Reads one expert with `pread`, looping because a short read is legal.
static int read_one(int fd, void *dst, size_t stride, uint32_t expert) {
    unsigned char *out = (unsigned char *)dst;
    off_t base = (off_t)expert * (off_t)stride;
    size_t done = 0;
    while (done < stride) {
        ssize_t got = pread(fd, out + done, stride - done, base + (off_t)done);
        if (got > 0) {
            done += (size_t)got;
            continue;
        }
        if (got == 0) {
            return EIO;               // short file: the caller's offsets are wrong
        }
        if (errno == EINTR) {
            continue;
        }
        return errno;
    }
    return 0;
}

static void *worker_main(void *arg) {
    nvmai_expert_reader *r = (nvmai_expert_reader *)arg;
    // Each worker owns one descriptor, so index it by position in the pool.
    int slot = -1;
    pthread_mutex_lock(&r->lock);
    for (int i = 0; i < r->threads; ++i) {
        if (pthread_equal(r->workers[i], pthread_self())) {
            slot = i;
            break;
        }
    }
    pthread_mutex_unlock(&r->lock);
    if (slot < 0) {
        return NULL;
    }
    int fd = r->fds[slot];

    pthread_mutex_lock(&r->lock);
    for (;;) {
        while (!r->shutting_down && r->next_index >= r->count) {
            pthread_cond_wait(&r->work_ready, &r->lock);
        }
        if (r->shutting_down) {
            break;
        }
        size_t index = r->next_index++;
        uint32_t expert = r->expert_ids[index];
        void *dst = r->destinations[index];
        pthread_mutex_unlock(&r->lock);

        int rc = read_one(fd, dst, r->expert_stride, expert);

        pthread_mutex_lock(&r->lock);
        if (rc != 0 && r->first_errno == 0) {
            r->first_errno = rc;
        }
        if (--r->outstanding == 0) {
            pthread_cond_signal(&r->work_done);
        }
    }
    pthread_mutex_unlock(&r->lock);
    return NULL;
}

nvmai_expert_reader *nvmai_expert_reader_create(const char *path,
                                               size_t expert_stride,
                                               int threads,
                                               int bypass_cache,
                                               int *out_errno) {
    if (out_errno) {
        *out_errno = 0;
    }
    if (path == NULL || expert_stride == 0) {
        if (out_errno) { *out_errno = EINVAL; }
        return NULL;
    }
    if (threads < 1) { threads = 1; }
    if (threads > NVMAI_IO_MAX_THREADS) { threads = NVMAI_IO_MAX_THREADS; }

    nvmai_expert_reader *r = (nvmai_expert_reader *)calloc(1, sizeof(*r));
    if (r == NULL) {
        if (out_errno) { *out_errno = ENOMEM; }
        return NULL;
    }
    r->expert_stride = expert_stride;
    r->threads = threads;
    for (int i = 0; i < NVMAI_IO_MAX_THREADS; ++i) {
        r->fds[i] = -1;
    }

    // One descriptor per worker: pread does not use the shared offset, but
    // separate descriptors keep the kernel's per-fd state uncontended.
    for (int i = 0; i < threads; ++i) {
        r->fds[i] = open(path, O_RDONLY);
        if (r->fds[i] >= 0 && bypass_cache) {
            // Best-effort: a kernel that refuses F_NOCACHE costs footprint
            // predictability, not correctness, so it must not fail the open.
            (void)fcntl(r->fds[i], F_NOCACHE, 1);
        }
        if (r->fds[i] < 0) {
            int err = errno;
            for (int j = 0; j < i; ++j) { close(r->fds[j]); }
            free(r);
            if (out_errno) { *out_errno = err; }
            return NULL;
        }
    }

    if (pthread_mutex_init(&r->lock, NULL) != 0
        || pthread_cond_init(&r->work_ready, NULL) != 0
        || pthread_cond_init(&r->work_done, NULL) != 0) {
        for (int j = 0; j < threads; ++j) { close(r->fds[j]); }
        free(r);
        if (out_errno) { *out_errno = ENOMEM; }
        return NULL;
    }

    // Publish thread identities under the lock before any worker looks for its
    // own slot, otherwise a fast worker can fail to find itself.
    pthread_mutex_lock(&r->lock);
    int started = 0;
    for (int i = 0; i < threads; ++i) {
        if (pthread_create(&r->workers[i], NULL, worker_main, r) != 0) {
            break;
        }
        started++;
    }
    if (started < threads) {
        // Bring up whatever started, then fail: a partial pool would silently
        // read at a fraction of the requested rate.
        r->shutting_down = 1;
        r->threads = started;
        pthread_cond_broadcast(&r->work_ready);
        pthread_mutex_unlock(&r->lock);
        for (int i = 0; i < started; ++i) { pthread_join(r->workers[i], NULL); }
        for (int j = 0; j < threads; ++j) { close(r->fds[j]); }
        pthread_cond_destroy(&r->work_ready);
        pthread_cond_destroy(&r->work_done);
        pthread_mutex_destroy(&r->lock);
        free(r);
        if (out_errno) { *out_errno = EAGAIN; }
        return NULL;
    }
    pthread_mutex_unlock(&r->lock);
    return r;
}

void nvmai_expert_reader_destroy(nvmai_expert_reader *r) {
    if (r == NULL) {
        return;
    }
    pthread_mutex_lock(&r->lock);
    r->shutting_down = 1;
    pthread_cond_broadcast(&r->work_ready);
    pthread_mutex_unlock(&r->lock);
    for (int i = 0; i < r->threads; ++i) {
        pthread_join(r->workers[i], NULL);
    }
    for (int i = 0; i < NVMAI_IO_MAX_THREADS; ++i) {
        if (r->fds[i] >= 0) { close(r->fds[i]); }
    }
    pthread_cond_destroy(&r->work_ready);
    pthread_cond_destroy(&r->work_done);
    pthread_mutex_destroy(&r->lock);
    free(r);
}

int nvmai_expert_reader_fetch(nvmai_expert_reader *r,
                              const uint32_t *expert_ids,
                              void *const *destinations,
                              size_t count) {
    if (r == NULL || (count > 0 && (expert_ids == NULL || destinations == NULL))) {
        return EINVAL;
    }
    if (count == 0) {
        return 0;
    }

    pthread_mutex_lock(&r->lock);
    r->expert_ids = expert_ids;
    r->destinations = destinations;
    r->count = count;
    r->next_index = 0;
    r->outstanding = count;
    r->first_errno = 0;
    r->generation++;
    pthread_cond_broadcast(&r->work_ready);
    while (r->outstanding > 0) {
        pthread_cond_wait(&r->work_done, &r->lock);
    }
    int rc = r->first_errno;
    // Leave the batch empty so idle workers block instead of spinning on a
    // consumed batch.
    r->count = 0;
    r->next_index = 0;
    r->expert_ids = NULL;
    r->destinations = NULL;
    pthread_mutex_unlock(&r->lock);
    return rc;
}

int nvmai_expert_reader_threads(const nvmai_expert_reader *r) {
    return r == NULL ? 0 : r->threads;
}
