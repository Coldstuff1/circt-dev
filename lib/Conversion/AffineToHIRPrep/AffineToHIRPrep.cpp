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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace circt;
using namespace circt::hir;

namespace {

// Attribute names to strip from the converted IR.
static constexpr llvm::StringLiteral kHlsAttrsToStrip[] = {
    "hls.INTERFACE_LATENCY", "hls.INTERFACE_WR_LATENCY",
    "hls.INTERFACE_RD_LATENCY", "hls.INTERFACE_STORAGE_TYPE",
    "hls.INTERFACE_PORT",
};

/// Given the hir.memref.ports ArrayAttr, return the rd_latency of the first
/// read port, or -1 if there is none.
static int64_t getRdLatencyFromPorts(ArrayAttr ports) {
  if (!ports)
    return -1;
  for (auto portAttr : ports) {
    auto portDict = portAttr.dyn_cast<DictionaryAttr>();
    if (!portDict)
      continue;
    if (auto rdAttr = portDict.getAs<IntegerAttr>("rd_latency"))
      return rdAttr.getInt();
  }
  return -1;
}

struct AffineToHIRPrepPass : public AffineToHIRPrepBase<AffineToHIRPrepPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    // =========================================================================
    // Step 1: Process externally defined functions.
    //   - Rename hls.INTERFACE_LATENCY -> hir.delay on args/results.
    //   - Fix up argNames (preserve existing names, append "t").
    //   - Set resultNames to ["out"] if needed.
    //   - Collect them for FuncExternOp creation (Step 3).
    // =========================================================================
    module.walk([&](func::FuncOp func) {
      if (!func.isExternal())
        return;

      // Rename hls.INTERFACE_LATENCY -> hir.delay on arguments.
      for (unsigned i = 0; i < func.getNumArguments(); ++i) {
        if (auto latency = func.getArgAttrOfType<IntegerAttr>(
                i, "hls.INTERFACE_LATENCY")) {
          func.setArgAttr(i, "hir.delay", latency);
          func.removeArgAttr(i, StringAttr::get(func.getContext(), "hls.INTERFACE_LATENCY"));
        } else {
          func.setArgAttr(i, "hir.delay", builder.getI64IntegerAttr(0));
        }
      }

      // Rename hls.INTERFACE_LATENCY -> hir.delay on results.
      for (unsigned i = 0; i < func.getNumResults(); ++i) {
        if (auto latency = func.getResultAttrOfType<IntegerAttr>(
                i, "hls.INTERFACE_LATENCY")) {
          func.setResultAttr(i, "hir.delay", latency);
          func.removeResultAttr(i, StringAttr::get(func.getContext(), "hls.INTERFACE_LATENCY"));
        } else {
          func.setResultAttr(i, "hir.delay", builder.getI64IntegerAttr(0));
        }
      }

      // Fix argNames: preserve existing names and append "t" for the time arg.
      // The existing argNames from Polygeist may already have the real names
      // (e.g. ["a", "b"]); we keep them and just append "t".
      SmallVector<Attribute> argNames;
      if (auto existingNames =
              func->getAttrOfType<ArrayAttr>("argNames")) {
        for (auto name : existingNames)
          argNames.push_back(name);
      } else {
        for (unsigned i = 0; i < func.getNumArguments(); ++i)
          argNames.push_back(
              builder.getStringAttr("arg" + std::to_string(i)));
      }
      argNames.push_back(builder.getStringAttr("t"));
      func->setAttr("argNames", builder.getArrayAttr(argNames));

      // Set result names to ["out"] for single-result functions.
      if (func.getNumResults() > 0)
        func->setAttr("resultNames",
                      builder.getArrayAttr(
                          {builder.getStringAttr("out")}));

    });

    // =========================================================================
    // Step 2: Process non-external (top-level) functions.
    // =========================================================================
    module.walk([&](func::FuncOp func) {
      if (func.isExternal())
        return;

      // -----------------------------------------------------------------------
      // 2a. Convert memref function arguments:
      //       hls.INTERFACE_WR_LATENCY / hls.INTERFACE_RD_LATENCY
      //       -> hir.memref.ports
      //     Convert scalar arguments:
      //       hls.INTERFACE_LATENCY -> hir.delay
      //     Strip all hls.* and llvm.linkage attrs afterwards.
      // -----------------------------------------------------------------------
      for (unsigned i = 0; i < func.getNumArguments(); ++i) {
        if (auto wrLatency = func.getArgAttrOfType<IntegerAttr>(
                i, "hls.INTERFACE_WR_LATENCY")) {
          DictionaryAttr dict = builder.getDictionaryAttr(
              {builder.getNamedAttr("wr_latency", wrLatency)});
          func.setArgAttr(i, "hir.memref.ports", builder.getArrayAttr({dict}));
        } else if (auto rdLatency = func.getArgAttrOfType<IntegerAttr>(
                       i, "hls.INTERFACE_RD_LATENCY")) {
          DictionaryAttr dict = builder.getDictionaryAttr(
              {builder.getNamedAttr("rd_latency", rdLatency)});
          func.setArgAttr(i, "hir.memref.ports", builder.getArrayAttr({dict}));
        } else if (auto latency = func.getArgAttrOfType<IntegerAttr>(
                       i, "hls.INTERFACE_LATENCY")) {
          func.setArgAttr(i, "hir.delay", latency);
        }
        // Strip all hls.* attrs from this argument.
        for (auto name : kHlsAttrsToStrip)
          func.removeArgAttr(i, StringAttr::get(func.getContext(), name));
      }

      // Strip llvm.linkage and any hls.* attrs from the function itself.
      func->removeAttr("llvm.linkage");
      func->removeAttr("resultNames");
      for (auto name : kHlsAttrsToStrip)
        func->removeAttr(name);

      // -----------------------------------------------------------------------
      // 2b. Add hwAccel attribute on all non-external functions.
      // -----------------------------------------------------------------------
      func->setAttr("hwAccel", builder.getUnitAttr());

      // -----------------------------------------------------------------------
      // 2c. Convert scalar memref.alloca (rank-0) to rank-1 memref<1xT>
      //     with mem_kind="reg" and hir.memref.ports = [{rd_latency=0},
      //     {wr_latency=1}].  Also erase any stores of llvm.mlir.undef
      //     into the new alloca (they are C-level uninitialised scalar vars
      //     that the HIR model doesn't need).
      // -----------------------------------------------------------------------
      SmallVector<memref::AllocaOp> allocasToConvert;
      func.walk([&](memref::AllocaOp alloca) {
        auto type = alloca.getType().cast<MemRefType>();
        if (type.getRank() == 0)
          allocasToConvert.push_back(alloca);
      });

      for (auto alloca : allocasToConvert) {
        auto type = alloca.getType().cast<MemRefType>();
        builder.setInsertionPoint(alloca);
        auto newType = MemRefType::get({1}, type.getElementType());
        auto newAlloca = builder.create<memref::AllocaOp>(alloca.getLoc(), newType);
        newAlloca->setAttr("mem_kind", builder.getStringAttr("reg"));

        DictionaryAttr regR = builder.getDictionaryAttr(
            {builder.getNamedAttr("rd_latency",
                                  builder.getI64IntegerAttr(0))});
        DictionaryAttr regW = builder.getDictionaryAttr(
            {builder.getNamedAttr("wr_latency",
                                  builder.getI64IntegerAttr(1))});
        newAlloca->setAttr("hir.memref.ports",
                           builder.getArrayAttr({regR, regW}));

        // Replace uses of 0-D memref with the new 1-D memref.
        alloca.replaceAllUsesWith(newAlloca.getResult());

        // Rewrite loads/stores: add explicit index 0.
        AffineMap zeroMap = builder.getConstantAffineMap(0);
        SmallVector<Operation *> users(newAlloca.getResult().getUsers().begin(),
                                       newAlloca.getResult().getUsers().end());
        SmallVector<Operation *> undefOpsToErase;
        for (auto *user : users) {
          if (auto load = dyn_cast<affine::AffineLoadOp>(user)) {
            if (load.getMemref() == newAlloca.getResult() &&
                load.getMap().getNumResults() == 0) {
              builder.setInsertionPoint(load);
              auto newLoad = builder.create<affine::AffineLoadOp>(
                  load.getLoc(), newAlloca.getResult(), zeroMap, ValueRange{});
              load.replaceAllUsesWith(newLoad.getResult());
              load.erase();
            }
          } else if (auto store = dyn_cast<affine::AffineStoreOp>(user)) {
            if (store.getMemref() == newAlloca.getResult() &&
                store.getMap().getNumResults() == 0) {
              // E: Erase stores of llvm.mlir.undef (spurious C-level
              // uninitialiser); keep real stores.
              Value val = store.getValueToStore();
              bool isUndef =
                  isa_and_nonnull<LLVM::UndefOp>(val.getDefiningOp());
              if (isUndef) {
                // Collect undef op for erasure after the store is gone.
                if (val.use_empty() || val.hasOneUse())
                  undefOpsToErase.push_back(val.getDefiningOp());
                store.erase();
              } else {
                builder.setInsertionPoint(store);
                builder.create<affine::AffineStoreOp>(
                    store.getLoc(), val, newAlloca.getResult(), zeroMap,
                    ValueRange{});
                store.erase();
              }
            }
          }
        }
        // Erase undef ops that are now unused.
        for (auto *op : undefOpsToErase)
          if (op->use_empty())
            op->erase();

        alloca.erase();
      }

      // -----------------------------------------------------------------------
      // 2d. Annotate affine.load ops with {result_delays=[rd_latency]}
      //     derived from the hir.memref.ports of the accessed memref.
      // -----------------------------------------------------------------------
      func.walk([&](affine::AffineLoadOp load) {
        Value memref = load.getMemref();
        ArrayAttr ports = nullptr;

        // Case 1: memref is a function argument.
        if (auto blockArg = memref.dyn_cast<BlockArgument>()) {
          unsigned argIdx = blockArg.getArgNumber();
          if (auto portsAttr = func.getArgAttrOfType<ArrayAttr>(
                  argIdx, "hir.memref.ports"))
            ports = portsAttr;
        }
        // Case 2: memref is defined by an AllocaOp.
        else if (auto allocaOp =
                     dyn_cast_or_null<memref::AllocaOp>(
                         memref.getDefiningOp())) {
          if (auto portsAttr =
                  allocaOp->getAttrOfType<ArrayAttr>("hir.memref.ports"))
            ports = portsAttr;
        }

        if (!ports)
          return;
        int64_t rdLatency = getRdLatencyFromPorts(ports);
        if (rdLatency < 0)
          return;
        load->setAttr(
            "result_delays",
            builder.getArrayAttr({builder.getI64IntegerAttr(rdLatency)}));
      });

      // -----------------------------------------------------------------------
      // 2e. Annotate func.call ops with {result_delays} from the callee's
      //     hir.delay on its result (item A: use i64).
      // -----------------------------------------------------------------------
      func.walk([&](func::CallOp call) {
        auto callee =
            module.lookupSymbol<func::FuncOp>(call.getCallee());
        if (!callee)
          return;
        int64_t delay = 0;
        if (callee.getNumResults() > 0) {
          if (auto attr = callee.getResultAttrOfType<IntegerAttr>(
                  0, "hir.delay"))
            delay = attr.getInt();
          else if (auto attr2 = callee.getResultAttrOfType<IntegerAttr>(
                       0, "hls.INTERFACE_LATENCY"))
            delay = attr2.getInt();
        }
        // Use I64 so the attr prints as bare [1] rather than [1 : i32].
        call->setAttr("result_delays",
                      builder.getArrayAttr(
                          {builder.getI64IntegerAttr(delay)}));
      });

      // -----------------------------------------------------------------------
      // 2f. Rename hls.PIPELINE_II -> II on affine.for loops.
      // -----------------------------------------------------------------------
      func.walk([&](affine::AffineForOp loop) {
        if (auto attr =
                loop->getAttrOfType<IntegerAttr>("hls.PIPELINE_II")) {
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
