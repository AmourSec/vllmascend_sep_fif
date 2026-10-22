#ifndef LIQ_HOST_SIMT_RUNTIME_H_
#define LIQ_HOST_SIMT_RUNTIME_H_

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <vector>

namespace liq_host_simt {

class Barrier {
public:
    void Configure(uint32_t participants) {
        std::lock_guard<std::mutex> lock(mutex_);
        participants_ = participants;
        arrived_ = 0;
        generation_ = 0;
    }

    void Wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        const uint64_t generation = generation_;
        if (++arrived_ == participants_) {
            arrived_ = 0;
            ++generation_;
            lock.unlock();
            condition_.notify_all();
            return;
        }
        condition_.wait(lock, [this, generation] { return generation_ != generation; });
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    uint32_t participants_ = 1;
    uint32_t arrived_ = 0;
    uint64_t generation_ = 0;
};

inline Barrier barrier;
inline std::mutex atomicMutex;
inline thread_local uint32_t thread = 0;
inline thread_local uint32_t threadCount = 1;
inline std::vector<int32_t> shuffleValues(2048, 0);

inline void Configure(uint32_t count) {
    barrier.Configure(count);
    threadCount = count;
}

inline void SetThread(uint32_t index) { thread = index; }

inline void ShuffleWrite(int32_t value) {
    shuffleValues.at(thread) = value;
    barrier.Wait();
}

inline int32_t ShuffleRead(uint32_t source) {
    const int32_t value = shuffleValues.at(source);
    barrier.Wait();
    return value;
}

}

#endif
