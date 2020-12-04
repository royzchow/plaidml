// Copyright 2020, Intel Corporation
#pragma once

#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Pass/Pass.h"

namespace pmlc::target::intel_level_zero {

#define GEN_PASS_CLASSES
#include "pmlc/target/intel_level_zero/passes.h.inc"

} // namespace pmlc::target::intel_level_zero
