// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "property_manager.hpp"

#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"

namespace ov {

class PropertyManagerImpl {
public:
    PropertyManagerImpl() {
        default_plugin_properties = {{ov::cache_dir.name(), ""},
                                     {ov::hint::allow_auto_batching.name(), ov::Any(true)},
                                     {ov::auto_batch_timeout.name(), "1000"}};
        default_global_properties = {{ov::force_tbb_terminate.name(), ov::Any(false)}};
    };
    ~PropertyManagerImpl(){};
    void merge_property(const ov::AnyMap& property, const std::string& plugin_name = {});

    ov::AnyMap exclude_property(const ov::AnyMap& property);

    ov::Any get_property(const std::string& property_name, const std::string& plugin_name = {});

    bool is_core_property(const std::string& property_name) {
        return (default_plugin_properties.count(property_name) > 0) ||
               (default_global_properties.count(property_name) > 0);
    }

private:
    void preprocess(ov::AnyMap& properties, const std::string& key, const std::string& value) {
        if (key == ov::cache_dir) {
            if (!value.empty()) {
                FileUtils::createDirectoryRecursive(value);
                properties[ov::cache_manager.name()] = std::make_shared<ov::FileStorageCacheManager>(value);
            }
        } else if (key == ov::force_tbb_terminate.name()) {
            auto flag = value == CONFIG_VALUE(YES) ? true : false;
            ov::threading::executor_manager()->set_property({{key, flag}});
        }
    }

    void print() {
        std::cout << "default_plugin_properties: " << std::endl;
        for (auto& item : default_plugin_properties) {
            std::cout << "    " << item.first << " : " << item.second.as<std::string>() << std::endl;
        }
        std::cout << "plugin_properties: " << std::endl;
        for (auto& item : plugin_properties) {
            std::cout << "    " << item.first << " :" << std::endl;
            for (auto& it : item.second) {
                std::cout << "        " << it.first << " : " << it.second.as<std::string>() << std::endl;
            }
        }
        std::cout << "default_global_properties: " << std::endl;
        for (auto& item : default_global_properties) {
            std::cout << "    " << item.first << " : " << item.second.as<std::string>() << std::endl;
        }
        std::cout << "internal_properties: " << std::endl;
        for (auto& item : internal_properties) {
            std::cout << "    " << item.first << " : " << item.second.as<std::string>() << std::endl;
        }
        std::cout << std::endl;
    }

private:
    mutable std::mutex mutex;
    // Default property can be used for all plugins, but its priority is low
    ov::AnyMap default_plugin_properties;

    // Plugin property is used for specified plugin, its priority is high
    // All models in a plugins will share the same plugin properties items
    // <device_name> : <ov::AnyMap>
    std::map<std::string, ov::AnyMap> plugin_properties;

    // Default global property will not be used for any plugin
    ov::AnyMap default_global_properties;

    // Internal properties
    ov::AnyMap internal_properties;
};

void PropertyManagerImpl::merge_property(const ov::AnyMap& properties, const std::string& plugin_name) {
    std::lock_guard<std::mutex> lock(mutex);
    auto merge = [&](ov::AnyMap& dst1, ov::AnyMap& dst2) {
        for (auto& it : default_plugin_properties) {
            auto item = properties.find(it.first);
            if (item != properties.end()) {
                dst1[item->first] = item->second;
                preprocess(dst2, item->first, item->second.as<std::string>());
            }
        }
    };
    if (!plugin_name.empty()) {
        ov::AnyMap value;
        auto _value = plugin_properties.find(plugin_name);
        if (_value != plugin_properties.end()) {
            value = _value->second;
        }
        merge(value, value);
        plugin_properties[plugin_name] = value;
    } else {
        merge(default_plugin_properties, internal_properties);
        merge(default_global_properties, internal_properties);
    }
    std::cout << "----------merge_property--------------" << std::endl;
    std::cout << "input properties for " << plugin_name << std::endl;
    for (auto& item : properties) {
        std::cout << "    " << item.first << " : " << item.second.as<std::string>() << std::endl;
    }
    print();
}

ov::AnyMap PropertyManagerImpl::exclude_property(const ov::AnyMap& property) {
    auto res = property;

    for (auto& it : default_plugin_properties) {
        auto item = res.find(it.first);
        if (item != res.end()) {
            res.erase(item);
        }
    }
    for (auto& it : default_global_properties) {
        auto item = res.find(it.first);
        if (item != res.end()) {
            res.erase(item);
        }
    }
    return res;
}

ov::Any PropertyManagerImpl::get_property(const std::string& name, const std::string& plugin_name) {
    std::cout << "----------get_property--------------" << std::endl;
    std::cout << "    plugin_name: " << plugin_name << "property_name: " << name << std::endl;
    print();

    std::lock_guard<std::mutex> lock(mutex);
    if (name == ov::force_tbb_terminate.name()) {
        const auto flag = ov::threading::executor_manager()->get_property(name).as<bool>();
        return flag ? CONFIG_VALUE(YES) : CONFIG_VALUE(NO);
    }

    if (!plugin_name.empty()) {
        ov::AnyMap value;
        auto _value = plugin_properties.find(plugin_name);
        if (_value != plugin_properties.end()) {
            value = _value->second;
            if (value.count(name) > 0) {
                return value.at(name);
            }
        }
    }

    if (default_plugin_properties.count(name) > 0) {
        return default_plugin_properties.at(name);
    }

    if (default_global_properties.count(name) > 0) {
        return default_global_properties.at(name);
    }

    if (internal_properties.count(name) > 0) {
        return internal_properties.at(name);
    }

    return ov::Any();
}

#define OV_CONFIG_MANAGER_CALL_STATEMENT(...)           \
    try {                                               \
        __VA_ARGS__;                                    \
    } catch (const std::exception& ex) {                \
        throw ov::Exception(ex.what());                 \
    } catch (...) {                                     \
        OPENVINO_ASSERT(false, "Unexpected exception"); \
    }

class PropertyManager::Impl : public PropertyManagerImpl {
public:
    Impl() : ov::PropertyManagerImpl() {}
};

PropertyManager::PropertyManager() {
    _impl = std::make_shared<Impl>();
}

void PropertyManager::merge_property(const ov::AnyMap& property, const std::string& plugin_name) {
    OV_CONFIG_MANAGER_CALL_STATEMENT(_impl->merge_property(property, plugin_name));
}

ov::AnyMap PropertyManager::exclude_property(const ov::AnyMap& property) {
    OV_CONFIG_MANAGER_CALL_STATEMENT(return _impl->exclude_property(property));
}

ov::Any PropertyManager::get_property(const std::string& name, const std::string& plugin_name) {
    OV_CONFIG_MANAGER_CALL_STATEMENT(return _impl->get_property(name, plugin_name));
}

bool PropertyManager::is_core_property(const std::string& property_name) {
    OV_CONFIG_MANAGER_CALL_STATEMENT(return _impl->is_core_property(property_name));
}

}  // namespace ov