//===- AffineToHIRPrep.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the AffineToHIRPrep pass.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_AFFINETOHIRPREP_H
#define CIRCT_CONVERSION_AFFINETOHIRPREP_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace circt {
std::unique_ptr<mlir::Pass> createAffineToHIRPrepPass();
} // namespace circt

#endif // CIRCT_CONVERSION_AFFINETOHIRPREP_H
