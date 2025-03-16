#ifndef TYPES_H
#define TYPES_H

#pragma once

#include <string>

namespace txeo {

/**
 * @brief Bundle of device information
 *
 */
struct DeviceInfo {
    /**
     * @brief Device name
     *
     */
    std::string name{};

    /**
     * @brief Device type (CPU or GPU)
     *
     */
    std::string device_type{};

    /**
     * @brief Memory limit in bytes
     *
     */
    size_t memory_limit{};
};

/**
 * @brief Normalization types to be used in normalization functions
 *
 */
enum class NormalizationType { MIN_MAX, Z_SCORE };

} // namespace txeo

#endif