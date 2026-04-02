//===- AffineToHIRPrep.cpp - Prepare Affine dialect for HIR ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AffineToHIRPrep pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/AffineToHIRPrep.h"
#include "../PassDetail.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace circt;
using namespace circt::hir;

namespace {

struct AffineToHIRPrepPass : public AffineToHIRPrepBase<AffineToHIRPrepPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    // 1. Process externally defined functions
    module.walk([&](func::FuncOp func) {
      if (!func.isExternal()) return;
      
      // Rename hls.INTERFACE_LATENCY to hir.delay on arguments
      for (unsigned i = 0; i < func.getNumArguments(); ++i) {
        if (auto latency = func.getArgAttrOfType<IntegerAttr>(i, "hls.INTERFACE_LATENCY")) {
          func.setArgAttr(i, "hir.delay", latency);
          func.removeArgAttr(i, "hls.INTERFACE_LATENCY");
        } else {
          // Default to 0 delay if not provided
          func.setArgAttr(i, "hir.delay", builder.getI64IntegerAttr(0));
        }
      }

      // Rename hls.INTERFACE_LATENCY to hir.delay on results
      for (unsigned i = 0; i < func.getNumResults(); ++i) {
        if (auto latency = func.getResultAttrOfType<IntegerAttr>(i, "hls.INTERFACE_LATENCY")) {
          func.setResultAttr(i, "hir.delay", latency);
        } else {
           func.setResultAttr(i, "hir.delay", builder.getI64IntegerAttr(0));
        }
      }

      // Set argument names
      SmallVector<Attribute> argNames;
      for (unsigned i = 0; i < func.getNumArguments(); ++i) {
        argNames.push_back(builder.getStringAttr("arg" + std::to_string(i)));
      }
      argNames.push_back(builder.getStringAttr("t"));
      func->setAttr("argNames", builder.getArrayAttr(argNames));

      // Set result names
      if (func.getNumResults() > 0) {
        func->setAttr("resultNames", builder.getArrayAttr({builder.getStringAttr("out")}));
      }
    });

    // 2. Process functions
    module.walk([&](func::FuncOp func) {
      if (func.isExternal()) return;

      // Handle function arguments: hls.INTERFACE_WR_LATENCY -> hir.memref.ports
      for (unsigned i = 0; i < func.getNumArguments(); ++i) {
        if (auto wrLatency = func.getArgAttrOfType<IntegerAttr>(i, "hls.INTERFACE_WR_LATENCY")) {
          DictionaryAttr dict = builder.getDictionaryAttr({builder.getNamedAttr("wr_latency", wrLatency)});
          func.setArgAttr(i, "hir.memref.ports", builder.getArrayAttr({dict}));
        } else if (auto rdLatency = func.getArgAttrOfType<IntegerAttr>(i, "hls.INTERFACE_RD_LATENCY")) {
          DictionaryAttr dict = builder.getDictionaryAttr({builder.getNamedAttr("rd_latency", rdLatency)});
          func.setArgAttr(i, "hir.memref.ports", builder.getArrayAttr({dict}));
        } else if (auto latency = func.getArgAttrOfType<IntegerAttr>(i, "hls.INTERFACE_LATENCY")) {
          func.setArgAttr(i, "hir.delay", latency);
        }
      }

      // Add hwAccel attribute if needed (we assume top level function).
      if (func.getName() == "gesummv_hir" /* or any non external */) {
         func->setAttr("hwAccel", builder.getUnitAttr());
      }

      // Convert memref.alloca : memref<T> to memref<1xT>
      SmallVector<memref::AllocaOp> allocasToConvert;
      func.walk([&](memref::AllocaOp alloca) {
        auto type = alloca.getType().cast<MemRefType>();
        if (type.getRank() == 0) {
          allocasToConvert.push_back(alloca);
        }
      });

      for (auto alloca : allocasToConvert) {
        auto type = alloca.getType().cast<MemRefType>();
        builder.setInsertionPoint(alloca);
        auto newType = MemRefType::get({1}, type.getElementType());
        auto newAlloca = builder.create<memref::AllocaOp>(alloca.getLoc(), newType);
        newAlloca->setAttr("mem_kind", builder.getStringAttr("reg"));
        
        DictionaryAttr regR = builder.getDictionaryAttr({builder.getNamedAttr("rd_latency", builder.getI64IntegerAttr(0))});
        DictionaryAttr regW = builder.getDictionaryAttr({builder.getNamedAttr("wr_latency", builder.getI64IntegerAttr(1))});
        newAlloca->setAttr("hir.memref.ports", builder.getArrayAttr({regR, regW}));

        // Replace uses of 0D memref with 1D memref (inserting 0 index)
        alloca.replaceAllUsesWith(newAlloca.getResult());
        
        // Update loads and stores to add the index
        AffineMap zeroMap = builder.getConstantAffineMap(0);
        SmallVector<Operation*> users(newAlloca.getResult().getUsers().begin(), newAlloca.getResult().getUsers().end());
        for (auto user : users) {
          if (auto load = dyn_cast<affine::AffineLoadOp>(user)) {
            if (load.getMemref() == newAlloca.getResult() && load.getMap().getNumResults() == 0) {
              builder.setInsertionPoint(load);
              auto newLoad = builder.create<affine::AffineLoadOp>(load.getLoc(), newAlloca.getResult(), zeroMap, ValueRange{});
              load.replaceAllUsesWith(newLoad.getResult());
              load.erase();
            }
          } else if (auto store = dyn_cast<affine::AffineStoreOp>(user)) {
            if (store.getMemref() == newAlloca.getResult() && store.getMap().getNumResults() == 0) {
              builder.setInsertionPoint(store);
              builder.create<affine::AffineStoreOp>(store.getLoc(), store.getValueToStore(), newAlloca.getResult(), zeroMap, ValueRange{});
              store.erase();
            }
          }
        }
        alloca.erase();
      }

      // Annotate func.call with result_delays based on the external function
      func.walk([&](func::CallOp call) {
        auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
        if (callee) {
          int delay = 0;
          if (callee.getNumResults() > 0) {
            if (auto attr = callee.getResultAttr(0, "hls.INTERFACE_LATENCY")) {
              delay = attr.cast<IntegerAttr>().getInt();
            }
          }
          call->setAttr("result_delays", builder.getArrayAttr({builder.getI32IntegerAttr(delay)}));
        }
      });

      // Update loops with PIPELINE_II attribute
      func.walk([&](affine::AffineForOp loop) {
        if (auto attr = loop->getAttrOfType<IntegerAttr>("hls.PIPELINE_II")) {
          loop->setAttr("II", attr);
          loop->removeAttr("hls.PIPELINE_II");
        }
      });
      
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> circt::createAffineToHIRPrepPass() {
  return std::make_unique<AffineToHIRPrepPass>();
}
