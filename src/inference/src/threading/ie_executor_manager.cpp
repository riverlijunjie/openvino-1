// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_parallel.hpp"
#if IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO
#    include <tbb/task_scheduler_init.h>
#endif

#include <memory>
#include <mutex>
#include <string>
#include <threading/ie_cpu_streams_executor.hpp>
#include <threading/ie_executor_manager.hpp>
#include <utility>

namespace InferenceEngine {
namespace {
class ExecutorManagerImpl : public ExecutorManager {
public:
    ~ExecutorManagerImpl();
    ITaskExecutor::Ptr getExecutor(const std::string& id) override;
    IStreamsExecutor::Ptr getIdleCPUStreamsExecutor(const IStreamsExecutor::Config& config) override;
    size_t getExecutorsNumber() const override;
    size_t getIdleCPUStreamsExecutorsNumber() const override;
    void clear(const std::string& id = {}) override;
    void setTbbFlag(bool flag) override;

private:
    bool tbbTerminateFlag = false;
    std::unordered_map<std::string, ITaskExecutor::Ptr> executors;
    std::vector<std::pair<IStreamsExecutor::Config, IStreamsExecutor::Ptr>> cpuStreamsExecutors;
    mutable std::mutex streamExecutorMutex;
    mutable std::mutex taskExecutorMutex;
    mutable std::mutex tbbMutex;
#if IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO
    std::shared_ptr<tbb::task_scheduler_init> _tbb = nullptr;
#endif
};

}  // namespace

void ExecutorManagerImpl::setTbbFlag(bool flag) {
    std::lock_guard<std::mutex> guard(tbbMutex);
    tbbTerminateFlag = flag;
#if IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO
    if (tbbTerminateFlag) {
        if (!_tbb) {
            _tbb = std::make_shared<tbb::task_scheduler_init>();
        }
    } else {
        _tbb = nullptr;
    }
#endif
}

ExecutorManagerImpl::~ExecutorManagerImpl() {
    clear();
    std::lock_guard<std::mutex> guard(tbbMutex);
    if (tbbTerminateFlag) {
#if IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO
        try {
            if (_tbb) {
                _tbb->blocking_terminate();
                std::cout << "\ttbb::blocking_terminate() is called." << std::endl;
            }
            _tbb = nullptr;
        } catch (std::exception& e) {
        }
#endif
    }
}

ITaskExecutor::Ptr ExecutorManagerImpl::getExecutor(const std::string& id) {
    std::lock_guard<std::mutex> guard(taskExecutorMutex);
    auto foundEntry = executors.find(id);
    if (foundEntry == executors.end()) {
        auto newExec = std::make_shared<CPUStreamsExecutor>(IStreamsExecutor::Config{id});
        executors[id] = newExec;
        return newExec;
    }
    return foundEntry->second;
}

IStreamsExecutor::Ptr ExecutorManagerImpl::getIdleCPUStreamsExecutor(const IStreamsExecutor::Config& config) {
    std::lock_guard<std::mutex> guard(streamExecutorMutex);
    for (const auto& it : cpuStreamsExecutors) {
        const auto& executor = it.second;
        if (executor.use_count() != 1)
            continue;

        const auto& executorConfig = it.first;
        if (executorConfig._name == config._name && executorConfig._streams == config._streams &&
            executorConfig._threadsPerStream == config._threadsPerStream &&
            executorConfig._threadBindingType == config._threadBindingType &&
            executorConfig._threadBindingStep == config._threadBindingStep &&
            executorConfig._threadBindingOffset == config._threadBindingOffset)
            if (executorConfig._threadBindingType != IStreamsExecutor::ThreadBindingType::HYBRID_AWARE ||
                executorConfig._threadPreferredCoreType == config._threadPreferredCoreType)
                return executor;
    }
    auto newExec = std::make_shared<CPUStreamsExecutor>(config);
    cpuStreamsExecutors.emplace_back(std::make_pair(config, newExec));
    return newExec;
}

size_t ExecutorManagerImpl::getExecutorsNumber() const {
    std::lock_guard<std::mutex> guard(taskExecutorMutex);
    return executors.size();
}

size_t ExecutorManagerImpl::getIdleCPUStreamsExecutorsNumber() const {
    std::lock_guard<std::mutex> guard(streamExecutorMutex);
    return cpuStreamsExecutors.size();
}

void ExecutorManagerImpl::clear(const std::string& id) {
    std::lock_guard<std::mutex> stream_guard(streamExecutorMutex);
    std::lock_guard<std::mutex> task_guard(taskExecutorMutex);
    if (id.empty()) {
        executors.clear();
        cpuStreamsExecutors.clear();
    } else {
        executors.erase(id);
        cpuStreamsExecutors.erase(
            std::remove_if(cpuStreamsExecutors.begin(),
                           cpuStreamsExecutors.end(),
                           [&](const std::pair<IStreamsExecutor::Config, IStreamsExecutor::Ptr>& it) {
                               return it.first._name == id;
                           }),
            cpuStreamsExecutors.end());
    }
}

namespace {

class ExecutorManagerHolder {
    std::mutex _mutex;
    std::shared_ptr<ExecutorManager> _manager = nullptr;
    int32_t _refCount = 0;

    ExecutorManagerHolder(const ExecutorManagerHolder&) = delete;
    ExecutorManagerHolder& operator=(const ExecutorManagerHolder&) = delete;

public:
    ExecutorManagerHolder() = default;
    ~ExecutorManagerHolder() = default;

    ExecutorManager::Ptr get(bool addRef) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_manager) {
            _manager = std::make_shared<ExecutorManagerImpl>();
            _refCount = 0;
        }
        if (addRef) {
            _refCount++;
        }
        return _manager;
    }

    void unref() {
        std::lock_guard<std::mutex> lock(_mutex);
        _refCount--;
        if (_refCount <= 0) {
            _manager = nullptr;
            _refCount = 0;
        }
    }
};

ExecutorManagerHolder& executorManagerHolder() {
    static ExecutorManagerHolder executorManagerHolder;
    return executorManagerHolder;
}

}  // namespace

void resetExecutorManager() {
    executorManagerHolder().unref();
}

ExecutorManager::Ptr executorManager(bool addRef) {
    return executorManagerHolder().get(addRef);
}

ExecutorManager* ExecutorManager::getInstance() {
    static auto ptr = executorManager().get();
    return ptr;
}

}  // namespace InferenceEngine
