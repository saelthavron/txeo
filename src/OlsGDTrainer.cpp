#include "txeo/OlsGDTrainer.h"

namespace txeo {

template <typename T>
void OlsGDTrainer<T>::set_learning_rate(double learning_rate) {
  this->reset();
  _learning_rate = learning_rate;
}

template <typename T>
double OlsGDTrainer<T>::learning_rate() const {
  return _learning_rate;
}

} // namespace txeo
