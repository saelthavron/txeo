#include "txeo/OrdLeastSquaresTrainer.h"

namespace txeo {

template <typename T>
void OrdLeastSquaresTrainer<T>::set_learning_rate(double learning_rate) {
  this->reset();
  _learning_rate = learning_rate;
}

template <typename T>
double OrdLeastSquaresTrainer<T>::learning_rate() const {
  return _learning_rate;
}

} // namespace txeo
