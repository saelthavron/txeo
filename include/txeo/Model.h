#ifndef MODEL_H
#define MODEL_H
#pragma once

#include <memory>

namespace txeo {

template <typename T>
class Model {
  private:
    struct Impl;
    std::unique_ptr<Impl> _impl{nullptr};

  public:
};

} // namespace txeo
#endif