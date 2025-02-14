#ifndef PREDICTOR_PRIVATE_H
#define PREDICTOR_PRIVATE_H
#pragma once

#include "txeo/Predictor.h"

#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/core/public/session.h>

template <typename T>
struct txeo::Predictor<T>::Impl {
    tensorflow::SavedModelBundle model;
    tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;

    Predictor<T>::TensorInfo in_name_shape_map;
    Predictor<T>::TensorInfo out_name_shape_map;
};

#endif