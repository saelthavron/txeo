#ifndef MODELPRIVATE_H
#define MODELPRIVATE_H
#pragma once

#include "txeo/Model.h"

#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/core/public/session.h>

template <typename T>
struct txeo::Model<T>::Impl {
    tensorflow::SavedModelBundle model;
    tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;
};

#endif