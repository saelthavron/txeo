#ifndef ORDLEASTSQUARESTRAINER_H
#define ORDLEASTSQUARESTRAINER_H
#pragma once

#include "Trainer.h"

namespace txeo {

template <typename T>
class OrdLeastSquaresTrainer : public txeo::Trainer<T> {
  public:
    OrdLeastSquaresTrainer();
    OrdLeastSquaresTrainer(const OrdLeastSquaresTrainer &) = delete;
    OrdLeastSquaresTrainer(OrdLeastSquaresTrainer &&) = delete;
    OrdLeastSquaresTrainer &operator=(const OrdLeastSquaresTrainer &) = delete;
    OrdLeastSquaresTrainer &operator=(OrdLeastSquaresTrainer &&) = delete;
    ~OrdLeastSquaresTrainer();

  private:
};
;

} // namespace txeo

#endif