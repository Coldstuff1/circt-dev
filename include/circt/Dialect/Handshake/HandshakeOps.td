//===- Ops.td - Handshake operation definitions ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//============================================================================//
//
// This file includes modifications made as part of the Dynamatic project.
// See https://github.com/EPFL-LAP/dynamatic.
//
//===----------------------------------------------------------------------===//
//
// This file define Handshake ops in tablegen.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "circt/Dialect/Handshake/HandshakeAttributes.td"

// @mortbopet: some kind of support for interfaces as parent ops is currently
// being tracked here: https://github.com/llvm/llvm-project/pull/66196
class HasParentInterface<string interface>
    : ParamNativeOpTrait<"HasParentInterface", interface>, StructuralOpTrait {}

// Base class for Handshake dialect ops.
class Handshake_Op<string mnemonic, list<Trait> traits = []>
    : Op<Handshake_Dialect, mnemonic,
         traits #[HasParentInterface<"FineGrainedDataflowRegionOpInterface">,
         DeclareOpInterfaceMethods<NamedIOInterface>,
         DeclareOpInterfaceMethods<ControlInterface>]> {
}

// This is almost exactly like a standard FuncOp, except that it has some
// extra verification conditions.  In particular, each Value must
// only have a single use.  Also, it defines a Dominance-Free Scope
def FuncOp : Op<Handshake_Dialect, "func", [
   IsolatedFromAbove,
   FunctionOpInterface,
   Symbol,
   RegionKindInterface,
   OpAsmOpInterface,
   HasClock,
   FineGrainedDataflowRegionOpInterface
]> {
  let summary = "Handshake dialect function.";
  let description = [{
    The func operation represents a handshaked function.
    This is almost exactly like a standard FuncOp, except that it has
    some extra verification conditions. In particular, each Value must
    only have a single use.
  }];

  let arguments = (ins
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let results = (outs);
  let regions = (region AnyRegion : $body);

  let skipDefaultBuilders = 1;

  let builders =
       [OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
                      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];

  let extraClassDeclaration = [{
    // Add an entry block to an empty function, and set up the block arguments
    // to match the signature of the function.
    Block *addEntryBlock();

    /// Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }

    /// Returns the region on the current operation that is callable.
    ::mlir::Region *getCallableRegion() {
      return &getBody();
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the names of the arguments to this function.
    ArrayAttr getArgNames() {
      return (*this)->getAttrOfType<ArrayAttr>("argNames");
    }

    /// Returns the names of the results from this function.
    ArrayAttr getResNames() {
      return (*this)->getAttrOfType<ArrayAttr>("resNames");
    }

    /// Returns the argument name at the given index.
    StringAttr getArgName(unsigned idx) {
      return getArgNames()[idx].cast<StringAttr>();
    }

    /// Returns the result name at the given index.
    StringAttr getResName(unsigned idx) {
      return getResNames()[idx].cast<StringAttr>();
    }

    /// Resolve argument and result names. This can be used during building of
    /// a handshake.func operation to ensure that names provided by an incoming
    /// operation fulfills the name requirements of the handshake function.
    void resolveArgAndResNames();

    /// Hook for FunctionOpInterface, called after verifying that the 'type'
    /// attribute is present and checks if it holds a function type.  Ensures
    /// getType, getNumFuncArguments, and getNumFuncResults can be called
    /// safely.
    LogicalResult verifyType() {
      auto type = getFunctionTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError(
            "requires '" + getFunctionTypeAttrName().getValue() +
            "' attribute of function type");
      return success();
    }

    /// Returns the body block of the function.
    Block* getBodyBlock() {
      return &getBody().front();
    }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    static ::llvm::StringRef getDefaultDialect() {
      return "handshake";
    }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

// InstanceOp
def InstanceOp : Handshake_Op<"instance", [
    CallOpInterface,
    HasClock,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "module instantiate operation";
  let description = [{
    The `instance` operation represents the instantiation of a module.  This
    is similar to a function call, except that different instances of the
    same module are guaranteed to have their own distinct state.
    The instantiated module is encoded as a symbol reference attribute named
    "module". An instance operation takes a control input as its last argument
    and returns a control output as its last result.

    Example:
    ```mlir
    %2:2 = handshake.instance @my_add(%0, %1, %ctrl) : (f32, f32, none) -> (f32, none)
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$module, Variadic<AnyType>:$opOperands);
  let results = (outs Variadic<AnyType>);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    (ins "FuncOp":$module, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("module", SymbolRefAttr::get(module));
      $_state.addTypes(module.getResultTypes());
  }]>, OpBuilder<
    (ins "SymbolRefAttr":$module, "TypeRange":$results,
     CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("module", module);
      $_state.addTypes(results);
      $_state.addTypes({$_builder.getType<::mlir::NoneType>()});
  }]>, OpBuilder<
    (ins "StringRef":$module, "TypeRange":$results,
     CArg<"ValueRange", "{}">:$operands), [{
       build($_builder, $_state,
             SymbolRefAttr::get($_builder.getContext(), module), results,
             operands);
  }]>];

  let extraClassDeclaration = [{
    FunctionType getModuleType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the module of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("module");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr(getModuleAttrName(), callee.get<mlir::SymbolRefAttr>());
    }

    /// Get the control operand of this instance op
    Value getControl() {
      return getOperands().back();
    }

    MutableOperandRange getArgOperandsMutable() {
      return getOpOperandsMutable();
    }

  }];

  let assemblyFormat = [{
    $module `(` $opOperands `)` attr-dict `:` functional-type($opOperands, results)
  }];
}

// This is almost exactly like a standard FuncOp, except that it has some
// extra verification conditions.  In particular, each Value must
// only have a single use.
def ReturnOp : Handshake_Op<"return", [Terminator]> {
  let summary = "Handshake dialect return.";
  let description = [{
    The return operation represents a handshaked
    function.  This is almost exactly like a standard ReturnOp, except
    that it exists in a handshake.func.  It has the same operands as
    standard ReturnOp which it replaces and an additional control -
    only operand(exit point of control - only network).
  }];

  let arguments = (ins Variadic<AnyType> : $opOperands);

  let hasVerifier = 1;
  let assemblyFormat =
    [{  attr-dict ($opOperands^ `:` type($opOperands))? }];
}

// Here use I32EnumAttr to better suit the BufferOp sequential
// attribute for it can only be seq or fifo.
def BufferCaseSeq: I32EnumAttrCase<"seq", 0>;
def BufferCaseFIFO: I32EnumAttrCase<"fifo", 1>;

def BufferTypeEnum: I32EnumAttr<
  "BufferTypeEnum", "BufferOp seq or fifo", [BufferCaseSeq, BufferCaseFIFO]> {
  let genSpecializedAttr = 0;
}
def BufferTypeEnumAttr: EnumAttr<Handshake_Dialect, BufferTypeEnum, "buffer_type_enum">;

def BufferOp : Handshake_Op<"buffer", [Pure, HasClock, SameOperandsAndResultType,
  DeclareOpInterfaceMethods<SOSTInterface, ["getSize"]>,
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>]> {
  let summary = "buffer operation";

  let description = [{
    The buffer operation represents a buffer operation. $slots
    must be an unsigned integer larger than 0. $bufferType=BufferTypeEnum::seq indicates a
    nontransparent buffer, while $bufferType=BufferTypeEnum::fifo indicates a transparent
    buffer.

    An 'initValues' attribute containing a list of integer values may be provided.
    The list must be of the same length as the number of slots. This will
    initialize the buffer with the given values upon reset.
    For now, only sequential buffers are allowed to have initial values.
    @todo: How to support different init types? these have to be stored (and
    retrieved) as attributes, hence they must be of a known type.
  }];

  let arguments = (ins
    AnyType:$operand,
    ConfinedAttr<I32Attr, [IntMinValue<1>]>:$slots,
    BufferTypeEnumAttr:$bufferType,
    OptionalAttr<I64ArrayAttr>:$initValues);
  let results = (outs AnyType:$result);

  let builders = [OpBuilder<
      (ins "Value":$operand, "int":$slots, "BufferTypeEnum":$bufferType), [{
        $_state.addOperands(operand);
        $_state.addTypes(operand.getType());
        $_state.addAttribute("slots", $_builder.getI32IntegerAttr(slots));
        $_state.addAttribute("bufferType", BufferTypeEnumAttr::get($_state.getContext(), bufferType));
    }]>];

  let extraClassDeclaration = [{
    bool isSequential() {
      return this->getBufferType() == BufferTypeEnum::seq;
    }
    int getNumSlots() {
      return (*this)->getAttrOfType<IntegerAttr>("slots").getValue().getZExtValue();
    }
    SmallVector<int64_t> getInitValueArray() {
      assert(getInitValues() && "initValues attribute not set");
      SmallVector<int64_t> values;
      for (auto value : (*this)->getAttrOfType<ArrayAttr>("initValues"))
        values.push_back(value.cast<IntegerAttr>().getValue().getSExtValue());
      return values;
    }

  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def ForkOp : Handshake_Op<"fork", [
  Pure, HasClock, SameOperandsAndResultType,
  DeclareOpInterfaceMethods<SOSTInterface, ["getSize"]>, 
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>
]> {
  let summary = "fork operation";

  let description = [{
    The fork operation represents a fork operation.  A
    single input is replicated to N outputs and distributed to each
    output as soon as the corresponding successor is available.

    Example:
    ```mlir
    %1:2 = fork [2] %0 : i32
    ```
  }];

  let arguments = (ins AnyType : $operand);
  let results = (outs Variadic<AnyType> : $result);
  
  let builders = [OpBuilder<
    (ins "Value":$operand, "unsigned":$size), [{
      $_state.addOperands(operand);
      SmallVector<Type> resultTypes {size, operand.getType()};
      $_state.addTypes(resultTypes);
  }]>];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
}

def LazyForkOp : Handshake_Op<"lazy_fork", [
  Pure, SameOperandsAndResultType,
  DeclareOpInterfaceMethods<SOSTInterface, ["getSize", "sostIsControl"]>
]> {
  let summary = "lazy fork operation";
  let description = [{
    The lazy_fork operation represents a lazy fork operation.
    A single input is replicated to N outputs and distributed to each
    output when all successors are available.

    Example:
    ```mlir
    %1:2 = lazy_fork [2] %0 : i32
    ```
  }];

  let arguments = (ins AnyType : $operand);
  let results = (outs Variadic<AnyType> : $result);

  let builders = [OpBuilder<
    (ins "Value":$operand, "unsigned":$size), [{
      $_state.addOperands(operand);
      SmallVector<Type> resultTypes {size, operand.getType()};
      $_state.addTypes(resultTypes);
  }]>];

  let hasCustomAssemblyFormat = 1;
}

def MergeOp : Handshake_Op<"merge", [
  Pure, MergeLikeOpInterface, SOSTInterface, SameOperandsAndResultType,
  DeclareOpInterfaceMethods<ExecutableOpInterface>
]> {
  let summary = "merge operation";
  let description = [{
    The merge operation represents a (nondeterministic)
    merge operation. Any input is propagated to the single output. The
    number of inputs corresponds to the number of predecessor
    blocks.

    Example:
    ```
    %0 = merge %a, %b, %c : i32
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$dataOperands);
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasCanonicalizer = 1;
}

def MuxOp : Handshake_Op<"mux", [
  Pure, MergeLikeOpInterface,
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
  DeclareOpInterfaceMethods<ControlInterface, ["isControl"]>,
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>
]> {
  let summary = "mux operation";
  let description = [{
    The mux operation represents a(deterministic) merge operation.
    Operands: select, data0, data1, data2, ...

    The 'select' operand is received from ControlMerge of the same
    block and it represents the index of the data operand that the mux
    should propagate to its single output.  The number of data inputs
    corresponds to the number of predecessor blocks.

    The mux operation is intended solely for control+dataflow selection.
    For purely dataflow selection, use the 'select' operation instead.

    Example:
    ```mlir
    %0 = mux %select [%data0, %data1, %data2] {attributes}: index, i32
    ```
  }];
  let arguments = (ins AnyType : $selectOperand, Variadic<AnyType> : $dataOperands);
  let results = (outs AnyType : $result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def ControlMergeOp : Handshake_Op<"control_merge", [
  Pure, MergeLikeOpInterface, HasClock, SOSTInterface,
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>
]> {
  let summary = "control merge operation";
  let description = [{
    The control_merge operation represents a
    (nondeterministic) control merge.  Any input is propagated to the
    first output and the index of the propagated input is sent to the
    second output.  The number of inputs corresponds to the number of
    predecessor blocks.

    Example:
    ```
    %0, %idx = control_merge %a, %b, %c {attributes} : i32, index
    ```
  }];

  let arguments = (ins Variadic<AnyType> : $dataOperands);
  let results = (outs AnyType : $result, AnyType : $index);
  
  let builders = [OpBuilder<
    (ins "ValueRange":$operands), [{
      assert(!operands.empty() && "cmerge needs at least one operand");
      $_state.addOperands(operands);
      // By default, the index result has an Index type
      $_state.addTypes(ArrayRef<Type>{operands[0].getType(),
                                      $_builder.getIndexType()});
  }]>];
  
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def BranchOp : Handshake_Op<"br", [
  Pure, SameOperandsAndResultType,
  DeclareOpInterfaceMethods<SOSTInterface, ["sostIsControl"]>, 
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>,
  AllTypesMatch<["dataOperand", "dataResult"]>
  ]> {
  let summary = "branch operation";
  let description = [{
      The branch operation represents an unconditional
      branch.  The single data input is propagated to the single
      successor.  The input must be triggered by some predecessor to
      avoid continous triggering of a successor block.

      Example:
      ```mlir
      %1 = br %0 : i32
      ```
  }];
  let arguments = (ins AnyType : $dataOperand);
  let results = (outs AnyType : $dataResult);
  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
}

def ConditionalBranchOp : Handshake_Op<"cond_br", [
  Pure, DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
  DeclareOpInterfaceMethods<ControlInterface, ["isControl"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>,
  TypesMatchWith<"data operand type matches true branch result type",
                    "dataOperand", "trueResult", "$_self">,
  TypesMatchWith<"data operand type matches false branch result type",
                    "dataOperand", "falseResult", "$_self">
]> {
  let summary = "conditional branch operation";
  let description = [{
     The cbranch operation represents a conditional
     branch.  The data input is propagated to one of the two outputs
     based on the condition input.

     Example:
     ```mlir
     %true, %false = conditional_branch %cond, %data : i32
     ```
  }];

  let arguments = (ins I1 : $conditionOperand, AnyType : $dataOperand);
  let results = (outs AnyType : $trueResult, AnyType : $falseResult);
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    // These are the indices into the dests list.
    enum { trueIndex = 0, falseIndex = 1 };
  }];
}

def SinkOp : Handshake_Op<"sink", [
  SOSTInterface, DeclareOpInterfaceMethods<ExecutableOpInterface>
]> {
  let summary = "sink operation";
  let description = [{
    The sink operation discards any data that arrives at its
         input.The sink has no successors and it can continuously consume data.

    Example:
    ```mlir
    sink %data : i32
    ```
  }];
  
  let arguments = (ins AnyType:$operand);
  let hasCustomAssemblyFormat = 1;
}

def SourceOp : Handshake_Op<"source", [
  Pure, DeclareOpInterfaceMethods<SOSTInterface, ["getDataType", "getSize"]> 
]> {
  let summary = "source operation";
  let description = [{
    The source operation represents continuous token
    source.  The source continously sets a 'valid' signal which the
    successor can consume at any point in time.
  }];

  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
}

def NeverOp : Handshake_Op<"never", [Pure]> {
  let summary = "never operation";
  let description = [{
    The never operation represents disconnected data
    source. The source never sets any 'valid' signal which will
    never trigger the successor at any point in time.
  }];

  let results = (outs AnyType:$result);
  let assemblyFormat = [{ attr-dict `:` qualified(type($result))}];
}

def ConstantOp : Handshake_Op<"constant", [
  Pure, DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>
]> {
  let summary = "constant operation";
  let description = [{
    The const has a constant value. When triggered by its
    single `ctrl` input, it sends the constant value to its single
    successor.

    Example:
    ```mlir
    %0 = constant %ctrl {value = 42 : i32} : i32
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value, NoneType:$ctrl);
  let results = (outs AnyType : $result);
  let hasCanonicalizer = 1;
  let assemblyFormat = [{ $ctrl attr-dict `:` qualified(type($result))}];
  let hasVerifier = 1;
}

def MemRefTypeAttr : TypeAttrBase<"MemRefType", "memref type attribute">;
def MemoryOp : Handshake_Op<"memory", [
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<MemoryOpInterface, ["allocateMemory", 
    "getMemOperands", "getMemResults"]>, HasClock,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>
]> {
  let summary = "memory";
  let description = [{
    Each MemoryOp represents an independent memory or memory region (BRAM or external memory).
    It receives memory access requests from load and store operations. For every request,
    it returns data (for load) and a data-less token indicating completion.
    The memory op represents a flat, unidimensional memory.
    Operands: all stores (stdata1, staddr1, stdata2, staddr2, ...), then all loads (ldaddr1, ldaddr2,...)
    Outputs: all load outputs, ordered the same as
    load data (lddata1, lddata2, ...), followed by all none outputs,
    ordered as operands (stnone1, stnone2,...ldnone1, ldnone2,...)
  }];
  let arguments = (ins Variadic<AnyType> : $inputs,
                   I32Attr : $ldCount,
                   I32Attr : $stCount,
                   I32Attr : $id,
                   MemRefTypeAttr : $memRefType,
                   I1Attr : $lsq);
  let results = (outs Variadic<AnyType> : $outputs);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(
      ins "ValueRange":$operands, "int":$outputs, "int":$control_outputs, "bool":$lsq,
      "int":$id, "Value":$memref)>
  ];

  let extraClassDeclaration = [{
    llvm::SmallVector<handshake::MemLoadInterface> getLoadPorts();
    llvm::SmallVector<handshake::MemStoreInterface> getStorePorts();
  }];

  let assemblyFormat = "`[` `ld` `=` $ldCount `,` `st` `=`  $stCount `]` `(` $inputs `)` attr-dict `:` $memRefType `,` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}

def ExternalMemoryOp : Handshake_Op<"extmemory", [
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>
]> {
  let summary = "external memory";
  let description = [{
    An ExternalMemoryOp represents a wrapper around a memref input to a
    handshake function. The semantics of the load/store operands are identical
    to what is decribed for MemoryOp. The only difference is that the first
    operand to this operand is a `memref` value.
    Upon lowering to FIRRTL, a handshake interface will be created in the
    top-level component for each load- and store which connected to this memory.

    Example:

    ```mlir
    handshake.func @main(%i: index, %v: i32, %mem : memref<10xi32>, %ctrl: none) -> none {
      %stCtrl = extmemory[ld = 0, st = 1](%mem : memref<10xi32>)(%vout, %addr) {id = 0 : i32} : (i32, index) -> (none)
      %vout, %addr = store(%v, %i, %ctrl) : (i32, index, none) -> (i32, index)
      ...
    }
    ```
  }];
  let arguments = (ins AnyMemRef:$memref,
                       Variadic<AnyType>:$inputs,
                       I32Attr : $ldCount,
                       I32Attr : $stCount,
                       I32Attr : $id);
  let results = (outs Variadic<AnyType>:$outputs);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(
      ins "Value":$memref, "ValueRange":$inputs, "int":$ldCount, "int":$stCount,
      "int":$id)>
  ];
  let assemblyFormat = "`[` `ld` `=` $ldCount `,` `st` `=`  $stCount `]` `(` $memref `:` qualified(type($memref)) `)` `(` $inputs `)` attr-dict `:` functional-type($inputs, $outputs)";

  let extraClassDeclaration = [{
    mlir::MemRefType getMemRefType() {
      return getMemref().getType().cast<mlir::MemRefType>();
    }

    llvm::SmallVector<handshake::MemLoadInterface> getLoadPorts();
    llvm::SmallVector<handshake::MemStoreInterface> getStorePorts();
  }];
}

def LoadOp : Handshake_Op<"load", [
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>
]> {
  let summary = "load operation";
  let description = [{
     Load memory port, sends load requests to MemoryOp. From dataflow
     predecessor, receives address indices and a control-only value
     which signals completion of all previous memory accesses which
     target the same memory.  When all inputs are received, the load
     sends the address indices to MemoryOp. When the MemoryOp returns
     a piece of data, the load sends it to its dataflow successor.

     Operands: address indices (from predecessor), data (from MemoryOp), control-only input.
     Results: data (to successor), address indices (to MemoryOp).

    Example:
    ```mlir
    %dataToSucc, %addr1ToMem, %addr2ToMem = load [%addr1, %addr2] %dataFromMem, %ctrl : i8, i16, index
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$addresses, AnyType:$data, NoneType:$ctrl);
  let results = (outs AnyType:$dataResult, Variadic<AnyType>:$addressResults);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$memref, "ValueRange":$indices)>];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def StoreOp : Handshake_Op<"store", [
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>
]> {
  let summary = "store operation";
  let description = [{
    Store memory port, sends store requests to MemoryOp. From dataflow
    predecessors, receives address indices, data, and a control-only
    value which signals completion of all previous memory accesses
    which target the same memory.  When all inputs are received, the
    store sends the address and data to MemoryOp.

    Operands: address indices, data, control-only input.
    Results: data and address indices (sent to MemoryOp).
    Types: data type followed by address type.

    Example:
    ```mlir
    %dataToMem, %addrToMem = store [%addr1, %addr2] %dataFromPred , %ctrl : i8, i16, index
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$addresses, AnyType:$data, NoneType:$ctrl);
  let results = (outs AnyType:$dataResult, Variadic<AnyType>:$addressResult);

  let builders =
      [OpBuilder<(ins "Value":$valueToStore, "ValueRange":$indices)>];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def JoinOp : Handshake_Op<"join", [
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<ControlInterface, ["isControl"]>,
  DeclareOpInterfaceMethods<GeneralOpInterface>
]> {
  let summary = "join operation";
  let description = [{
    A control-only synchronizer.  Produces a valid output when all
    inputs become available.

    Example:
    ```mlir
    %0 = join %a, %b, %c : i32, i1, none
    ```
  }];
  let arguments = (ins Variadic<AnyType> : $data);
  let results = (outs NoneType : $result);
  let hasCustomAssemblyFormat = 1;
}

def SyncOp : Handshake_Op<"sync", [
  HasClock, DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>,
  AllTypesMatch<["operands", "results"]>
]> {
  let summary = "sync operation";
  let description = [{
    Synchronizes an arbitrary set of inputs. Synchronization implies applying
    join semantics in between all in- and output ports.

    Example:
    ```mlir
    %aSynced, %bSynced, %cSynced = sync %a, %b, %c : i32, i1, none
    ```
  }];
  let arguments = (ins Variadic<AnyType> : $operands);
  let results = (outs Variadic<AnyType> : $results);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    (ins "ValueRange":$operands), [{
      $_state.addOperands(operands);
      $_state.addTypes(operands.getTypes());
  }]>];
  let assemblyFormat = [{ $operands attr-dict `:` type($operands)}];
}

def I4 : I<4>;
def I4Attr : SignlessIntegerAttrBase<I4, "4-bit integer attribute">;

def UnpackOp : Handshake_Op<"unpack", [
  HasClock, DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>,
  TypesMatchWith<"result types match element types of 'tuple'",
                     "input", "results",
                     "$_self.cast<TupleType>().getTypes()">
]> {
  let summary = "unpacks a tuple";
  let description = [{
    The `unpack` operation assigns each value of a tuple to a separate
    value for further processing. The number of results corresponds
    to the number of tuple elements.
    Similar to `fork`, each output is distributed as soon as the corresponding
    successor is ready.

    Example:
    ```mlir
    %a, %b = handshake.unpack %tuple {attributes} : tuple<i32, i64>
    ```
  }];

  let arguments = (ins Builtin_Tuple:$input);
  let results = (outs Variadic<AnyType>:$results);

  let builders = [OpBuilder<(ins "Value":$input), [{
    $_state.addOperands(input);

    TupleType type = input.getType().dyn_cast_or_null<TupleType>();
    assert(type && "expect unpack to have a 'TupleType' operand");
    $_state.addTypes(type.getTypes());
  }]>];

  let hasCustomAssemblyFormat = 1;
}

def PackOp : Handshake_Op<"pack", [
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>,
  TypesMatchWith<"input types match element types of 'tuple'",
                   "result", "inputs",
                   "$_self.cast<TupleType>().getTypes()">
]> {

  let summary = "packs a tuple";
  let description = [{
    The `pack` operation constructs a tuple from separate values.
    The number of operands corresponds to the number of tuple elements.
    Similar to `join`, the output is ready when all inputs are ready.

    Example:
    ```mlir
    %tuple = handshake.pack %a, %b {attributes} : tuple<i32, i64>
    ```
    }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Builtin_Tuple:$result);

  let builders = [OpBuilder<
    (ins "ValueRange":$operands), [{
      $_state.addOperands(operands);
      $_state.addTypes(TupleType::get($_builder.getContext(), operands.getTypes()));
  }]>];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Dynamatic operations
//===----------------------------------------------------------------------===//

def MemoryControllerOp : Handshake_Op<"mem_controller", [
  DeclareOpInterfaceMethods<MemoryOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>
]> {
  let summary = "memory controller (dynamatic)";
  let description = [{
    Each `MemoryControllerOp` represents an interface to an externally defined 
    unidimensional memory (i.e., it interfaces a memref input to a Handshake 
    function). It receives control signals from each basic block containing
    store operations referencing the wrapped memref; the formers are fed to the
    operation through constants indicating the number of stores the basic block
    will make to the referenced memory region. It also receives load (address
    value) and store (address value + data value) memory accesses, optionally
    fed throuhg an LSQ. It returns a data value for each load request as well as
    a single control signal to indicate basic block completion to the enclosing
    function's end operation.

    Additionally, the `connectedBlocks` attribute contains the ordered list of
    basic blocks (referenced by their unique IDs) that communicate with the
    controller. Every load/store operation is required to be tagged with the ID
    of a basic block present in this list.  

    Optionally, a memory controller may act as a middle-person between an LSQ
    and the external memory.

    The order of memory inputs is
    1. For each basic block:
      a. If there is at least one store in the block, a control signal fed
         through a constant indicating the number of store operations in the 
         block.
      b. Load/Store access requests from within the block, in program order.
    2. If an LSQ references the same memory region:
      a. Control signals from basic blocks containing store ports to the LSQ but
         not directly to the MC. These have the same semantics as the control
         signals described above.
      b. A single address operand for load accesses coming through the LSQ and
         a single pair of address/data operands for store accesses coming
         through the LSQ. 

    The order of memory outputs is
    1. Load results, in program order (i.e., in load accesses order).
    2. If an LSQ references the same memory region, a data result for load
       accesses.
    3. Control signal indicating completion.

    Example (without LSQ):
    ```mlir
    %ldData1, %ldData2, %ctrl = mem_controller[%mem : memref<16xi32>] (
        %bb0, %stAddr1, %stData1, %ldAddr1, %ldAddr2,
        %bb1, %stAddr2, %stData2
      ) {connectedBlocks = [0 : i32, 1 : i32]} :
      (i32, index, i32, index, index, i32, index, i32) -> (i32, i32, none)
    ```

    Example (with LSQ):
    ```mlir
    %ldData1, %ldData2, %ldDataToLSQ, %ctrl = mem_controller[memref<16xi32>] (
        %bb0, %stAddr1, %stData1, %ldAddr1, %ldAddr2,
        %bb1, %stAddr2, %stData2,
        %group3,
        %ldAddrFromLSQ, %stAddrFromLSQ, %stDataFromLSQ
      ) {connectedBlocks = [0 : i32, 1 : i32, 2 : i32]} :
      (i32, index, i32, index, index, i32, index, i32, i32, index, index, i32) ->
        (i32, i32, i32, none)
    ```
  }];
  
  let arguments = (ins  AnyMemRef : $memRef, Variadic<AnyType> : $memInputs,
                        I32ArrayAttr : $connectedBlocks);
  let results = (outs Variadic<AnyType> : $memOutputs, NoneType : $done);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins  "Value":$memRef, "ValueRange":$inputs,
                                  "ArrayRef<unsigned>":$blocks, 
                                  "unsigned":$numLoads)>];
  let hasVerifier = 1;
  let assemblyFormat = " `[` $memRef `:` qualified(type($memRef)) `]` `(` $memInputs `)` attr-dict `:` functional-type($memInputs, results)";

  let extraClassDeclaration = [{
    /// Returns the list of basic block IDs the MC is connected to.
    mlir::SmallVector<unsigned> getMCBlocks() {
      mlir::SmallVector<unsigned> blocks;
      for (mlir::Attribute attr : getConnectedBlocks())
        blocks.push_back(cast<IntegerAttr>(attr).getValue().getZExtValue());
      return blocks;
    }

    /// Returns a convenient data-structure to go over the controls and memory
    /// accesses that are connected to the memory controller. 
    dynamatic::MCPorts getPorts();

    /// Returns the type of the memref referenced by the memory interface.
    mlir::MemRefType getMemRefType() {
      return getMemRef().getType().cast<mlir::MemRefType>();
    }
  }];
}

def LSQOp : Handshake_Op<"lsq", [
  DeclareOpInterfaceMethods<MemoryOpInterface, ["getMemOperands"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>
]> {
  let summary = "load-store queue  (dynamatic)";
  let description = [{
    Each `LSQOp` represents an interface to an externally defined unidimensional
    memory. In terms of circuit connectivity and semantics it is quite similar
    to the `MemoryControllerOp` operation (see the latter's documentation), with
    some important differences outlined below.
    - LSQs work on a group basis instead of on a block basis. Basic blocks can
      be considered groups but the opposite is not generally true (i.e., groups
      are more general than blocks). A group must satisfy a "linear dominance"
      constraint, meaning that all the memory accesses it contains must be
      clearly program-orderable, and stored in the LSQ's inputs in that order.
      Additionally, two groups connected to the same LSQ cannot share the same
      control signal, which are determined by the program location of the first
      memory access of each group). 
    - Control signals are not fed through a constant and are instead data-less
      signals simply indicating that a new group has started.
    - To ensure proper identification of groups when analyzing the operation's
      operands, the `groupSizes` attribute contains the numberf of load/store
      ports in each LSQ group, in operand order.  
    
    In the absence of an MC on the same memory region, the LSQ talks directly to
    the external memory and its first memory input will be the Handhsake
    function's memref argument that it interfaces in the circuit. In the
    presence of an MC on the same memory region, all load/store requests to the
    LSQ are forwarded to the MC (which talks to the external memory) and the
    first memory input is a control signal.

    The order of memory inputs is
    1. Wrapped memref (optional, if not connected to an MC).
    2. For each LSQ group:
      a. A control signal indicating that the group of accesses has started.
      b. Load/Store access requests from within the block, in program order.
    3. If an MC references the same memory region, a single data operand for
       load accesses coming back from an external memory through the MC.

    The order of results is
    1. Load results, in program order (i.e., in load accesses order).
    2. If an MC references the same memory region, a single address operand for
       load accesses and a single pair of address/data operands for store
       accesses going to the external memory through the MC. 
    3. Control signal indicating completion.

    Example (without MC):
    ```mlir
    %ldData1, %ldData2, %ctrl = lsq[%mem : memref<16xi32>] (
        %group1, %stAddr1, %stData1, %ldAddr1, %ldAddr2,
        %group2, %stAddr2, %stData2
      ) {groupSizes = [3 : i32, 1 : i32]} :
      (none, index, i32, index, index, none, index, i32) -> (i32, i32, none)
    ```

    Example (with MC):
    ```mlir
    %ldData1, %ldData2, %ldAddrToMC, %stAddrToMC, %stDataToMC, %ctrl =
      lsq[MC] (
        %group1, %stAddr1, %stData1, %ldAddr1, %ldAddr2,
        %group2, %stAddr2, %stData2, %ldDataFromMC
      ) {groupSizes = [3 : i32, 1 : i32]} :
      (none, index, i32, index, index, none, index, i32, i32) ->
        (i32, i32, index, index, i32, none)
    ```
  }];
  
  let arguments = (ins Variadic<AnyType> : $memInputs, I32ArrayAttr : $groupSizes);
  let results = (outs Variadic<AnyType> : $memOutputs, NoneType : $done);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins  "Value":$memref, "ValueRange":$inputs, 
                    "ArrayRef<unsigned>":$groupSizes, "unsigned":$numLoads)>,
    OpBuilder<(ins  "::circt::handshake::MemoryControllerOp":$mcOp, 
                    "ValueRange":$inputs, "ArrayRef<unsigned>":$groupSizes, 
                    "unsigned":$numLoads)>,
  ];
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns the list of sizes of all LSQ groups.
    mlir::SmallVector<unsigned> getLSQGroupSizes() {
      mlir::SmallVector<unsigned> sizes;
      for (mlir::Attribute attr : getGroupSizes())
        sizes.push_back(cast<IntegerAttr>(attr).getValue().getZExtValue());
      return sizes;
    }

    /// Returns a convenient data-structure to go over the controls and memory
    /// accesses that are connected to the LSQ. 
    dynamatic::LSQPorts getPorts();
    
    /// Determines whether the LSQ is connected to an MC. 
    bool isConnectedToMC() {
      return !isa<mlir::MemRefType>(getMemInputs().front().getType());
    }

    /// Returns the MC connected to the LSQ, if it exists.
    handshake::MemoryControllerOp getConnectedMC();

    /// Returns the memref referenced by the LSQ.
    mlir::Value getMemRef();

    /// Returns the type of the memref referenced by the LSQ.
    mlir::MemRefType getMemRefType();
  }];
}

class Handshake_LoadOp<string mnemonic> : Handshake_Op<mnemonic, [
  LoadOpInterface,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>,
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
]> {
  let arguments = (ins AnyType:$address, AnyType:$data);
  let results = (outs AnyType:$addressResult, AnyType:$dataResult);
  let builders = [OpBuilder<(ins "MemRefType":$memrefType, "Value":$address)>];
  let assemblyFormat = " `[` $address `]` $data attr-dict `:` qualified(type($address)) `,` qualified(type($data)) ";

  let extraClassDefinition = [{
    std::string $cppClass::getOperandName(unsigned int idx) {
      return (idx == 0) ? "addrIn" : "dataFromMem";
    }

    std::string $cppClass::getResultName(unsigned int idx) {
      return (idx == 0) ? "addrOut" : "dataOut";
    }

    mlir::LogicalResult $cppClass::inferReturnTypes(
        mlir::MLIRContext *context, std::optional<Location> location,
        mlir::ValueRange operands, mlir:: DictionaryAttr attributes,
        mlir::OpaqueProperties properties, mlir::RegionRange regions,
        mlir::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
      auto oprdTypes = operands.getTypes();
      inferredReturnTypes.append(oprdTypes.begin(), oprdTypes.end());
      return success();
    }
  }];
}

def MCLoadOp : Handshake_LoadOp<"mc_load"> {
  let summary = "load operation for memory controller (MC)";
  let description = [{
    Represents a load memory port which sends load requests to a memory
    controller (`circt::handshake::MemoryControllerOp`). It receives an address
    from a dataflow predecessor and a data value from the memory interface that
    eventually holds the loaded data. It returns an address that is sent to the
    memory interface as well as a data value that is sent to its dataflow successor.

    The order of operands is
    1. address (from predecessor)
    2. data value (from memory interface)

    The order of results is
    1. address (to memory interface).
    2. data value (to successor)

    Example:
    ```mlir
    %addrToMem, %dataToSucc = mc_load [%addrFromPred] %dataFromMem : i32, index
    ```
  }];
}

def LSQLoadOp : Handshake_LoadOp<"lsq_load"> {
  let summary = "load operation for load-store queue (LSQ)";
  let description = [{
    Represents a load memory port which sends load requests to an LSQ
    (`circt::handshake::LSQOp`). It receives an address from a dataflow
    predecessor and a data value from the memory interface that eventually holds
    the loaded data. It returns an address that is sent to the memory interface
    as well as a data value that is sent to its dataflow successor.

    The order of operands is
    1. address (from predecessor)
    2. data value (from memory interface)

    The order of results is
    1. address (to memory interface).
    2. data value (to successor)

    Example:
    ```mlir
    %addrToMem, %dataToSucc = lsq_load [%addrFromPred] %dataFromMem : i32, index
    ```
  }];
}

class Handshake_StoreOp<string mnemonic> : Handshake_Op<mnemonic, [
  StoreOpInterface,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>,
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
]> {
  let arguments = (ins AnyType:$address, AnyType:$data);
  let results = (outs AnyType:$addressResult, AnyType:$dataResult);
  let assemblyFormat = " `[` $address `]` $data attr-dict `:` qualified(type($data)) `,` qualified(type($address)) ";

  let extraClassDefinition = [{
    std::string $cppClass::getOperandName(unsigned int idx) {
      return (idx == 0) ? "addrIn" : "dataIn";
    }

    std::string $cppClass::getResultName(unsigned int idx) {
      return (idx == 0) ? "addrOut" : "dataToMem";
    }

    mlir::LogicalResult $cppClass::inferReturnTypes(
        mlir::MLIRContext *context, std::optional<Location> location,
        mlir::ValueRange operands, mlir:: DictionaryAttr attributes,
        mlir::OpaqueProperties properties, mlir::RegionRange regions,
        mlir::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
      auto oprdTypes = operands.getTypes();
      inferredReturnTypes.append(oprdTypes.begin(), oprdTypes.end());
      return success();
    }
  }];
}

def MCStoreOp : Handshake_StoreOp<"mc_store"> {
  let summary = "store operation for memory controller (MC)";
  let description = [{
    Represents a store memory port which sends store requests to a memory
    controller (`circt::handshake::MemoryControllerOp`). It receives an address
    and a data value to store in memory from its dataflow predecessors. It
    returns the value to store and an address, both of which are sent to the
    memory interface.

    The order of operands is
    1. address (from predecessor)
    2. data value (from predecessor)

    The order of results is
    1. address (to memory interface).
    2. data value (to memory interface)

    Example:
    ```mlir
     %addrToMem, %dataToMem = mc_store [%addrFromPred] %dataFromPred : i32, index
    ```
  }];
}

def LSQStoreOp : Handshake_StoreOp<"lsq_store"> {
  let summary = "store operation for load-store queue (LSQ)";
  let description = [{
    Represents a store memory port which sends store requests to an LSQ
    (`circt::handshake::LSQOp`). It receives an address and a data value to
    store in memory from its dataflow predecessors. It returns the value to
    store and an address, both of which are sent to the memory interface.

    The order of operands is
    1. address (from predecessor)
    2. data value (from predecessor)

    The order of results is
    1. address (to memory interface).
    2. data value (to memory interface)

    Example:
    ```mlir
     %addrToMem, %dataToMem = lsq_store [%addrFromPred] %dataFromPred : i32, index
    ```
  }];
}

def DynamaticReturnOp : Handshake_Op<"d_return", [
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
]> {
  let summary = "handshake-level return (dynamatic)";
  let description = [{
    This return operation represents the handshake equivalent of a 
    func::ReturnOp (i.e., the latter is converted to the former one-to-one,
    preserving operands). This operation also outputs a list of values that
    correspond one-to-one to its own operands (i.e., same type and semantic
    value).

    Example:
    ```mlir
    %out1, %out2 = d_return %in1, %in2 : i32, i64
    ```
  }];

  let arguments = (ins Variadic<AnyType> : $inputs);
  let results = (outs Variadic<AnyType> : $outputs);
  let hasVerifier = 1;
  let assemblyFormat = [{ attr-dict ($inputs^ `:` type($inputs))? }];
}

def EndOp : Handshake_Op<"end", [Terminator]> {
  let summary = "function endpoint (dynamatic)";
  let description = [{
    This operation is the terminator of every handshake function in Dynamatic.
    Its operands are the merged results of all return operations in the function
    (or simply the forwarded results of a single return) followed by the output
    control signals of all memory interfaces in the function.

    Example:
    ```mlir
    end %res1, %res2, %ctrl1, %ctrl2 : i32, i64, none, none
    ```
  }];

  let arguments = (ins Variadic<AnyType> : $inputs);

  let hasVerifier = 1;
  let assemblyFormat = [{ attr-dict ($inputs^ `:` type($inputs))? }];
  let extraClassDeclaration = [{
    /// Returns the range of operands that are actual function return values.
    mlir::ValueRange getReturnValues();

    /// Returns the range of operands that are memory control signals.
    mlir::ValueRange getMemoryControls();
  }];
}

//===----------------------------------------------------------------------===//
// Speculation operations
//===----------------------------------------------------------------------===//

def SpeculatorOp : Handshake_Op<"speculator", [ 
  TypesMatchWith<"data operand type matches data result type",
                 "dataIn", "dataOut", "$_self">]> {
  let summary = "Central control unit of the speculative circuit.";
  let description = [{
    The speculator produces speculative tokens when real data has
    not yet arrived. The speculator is also responsible for deciding 
    the correctness of its speculation.

    It needs to send control tokens to save units and commit units
    ($saveCtrl, $commitCtrl). For save-commit units, three control
    signals are needed. $SCSaveCtrl is connected directly, while 
    $SCCommitCtrl replicates the branches that $dataOut
    follows and might not reach a save-commit. $SCBranchCtrl is needed
    in the replicated branches to discard the latter signal.

    Example:
    ```mlir
    %dataOut, %saveCtrl, %commitCtrl, %SCSaveCtrl, 
      %SCCommitCtrl, %SCBranchCtrl = speculator %dataIn : i1
    ```
  }];

  let arguments = (ins AnyType : $dataIn);
  let results = (outs AnyType : $dataOut,
                  I1 : $saveCtrl, I1 : $commitCtrl, 
                  I<3> : $SCSaveCtrl, I<3> : $SCCommitCtrl, 
                  I1 : $SCBranchCtrl);
  let assemblyFormat = "$dataIn attr-dict `:` qualified(type($dataIn)) ";
}

def SpecSaveOp : Handshake_Op<"spec_save", [
  TypesMatchWith<"data operand type matches data result type",
                 "dataIn", "dataOut", "$_self">] > {
  let summary = "Saves data tokens that interact in the speculative region.";
  let description = [{
    Save units are used to mark the beginning of the speculation region. 
    Whenever a speculative token can interact with a non-speculative token, 
    the non-speculative token needs to be saved. In case of correct speculation
    or no speculation, the saved token can be dropped.

    On the other hand, if the speculation was incorrect, the saved tokens 
    need to be reinserted into the circuit to repeat the previously 
    miscalculated computations.

    Example:
    ```mlir
    %dataOut = spec_save[%ctrl] %dataIn : i11
    ```
  }];

  let arguments = (ins AnyType : $dataIn, I1 : $ctrl);
  let results = (outs AnyType : $dataOut);
  let assemblyFormat = " `[` $ctrl `]` $dataIn attr-dict `:` qualified(type($dataIn)) ";
}

def SpecCommitOp : Handshake_Op<"spec_commit", [
  TypesMatchWith<"data operand type matches data result type",
                    "dataIn", "dataOut", "$_self">] > {
  let summary = "Stall speculative data tokens until they are resolved.";
  let description = [{
    Commit units are used to mark the end of the speculation region. 
    Commit units are used to stall speculative tokens until they receive 
    the decision from the speculator. In case the speculation is correct, 
    the speculative tokens are converted into non-speculative tokens and 
    passed on to the rest of the circuit. 
    Otherwise, the speculative tokens are discarded. 

    Any non-speculative token can pass through the commit units 
    without any stall.

    Example:
    ```mlir
    %dataOut = spec_commit[%ctrl] %dataIn : i11
    ```
  }];

  let arguments = (ins AnyType : $dataIn, I1 : $ctrl);
  let results = (outs AnyType : $dataOut);
  let assemblyFormat = " `[` $ctrl `]` $dataIn attr-dict `:` qualified(type($dataIn)) ";
}

def SpecSaveCommitOp : Handshake_Op<"spec_save_commit", [
  TypesMatchWith<"data operand type matches data result type",
                    "dataIn", "dataOut", "$_self">]> {
  let summary = "Lets all tokens pass and saves a copy of them.";
  let description = [{
    Save-Commits are used when speculation occurs in a loop.
    To increase loop parallelism, the save-commit unit will let both
    speculative and non-speculative tokens pass as well as save
    a copy of them.
 
    Example:
    ```mlir
    %dataOut = spec_save_commit[%ctrl] %dataIn : i11
    ```
  }];

  let arguments = (ins AnyType : $dataIn, I<3> : $ctrl);
  let results = (outs AnyType : $dataOut);
  let assemblyFormat = " `[` $ctrl `]` $dataIn attr-dict `:` qualified(type($dataIn)) ";
}

def SpeculatingBranchOp : Handshake_Op<"speculating_branch", [
  TypesMatchWith<"data operand type matches true branch result type",
                    "dataOperand", "trueResult", "$_self">,
  TypesMatchWith<"data operand type matches false branch result type",
                    "dataOperand", "falseResult", "$_self">
]> {
  let summary = "speculating branch operation";
  let description = [{
     The speculating branch operation represents a conditional
     branch that decides the condition based on if a token Value is
     speculative or not. 

     The $tagFromOperand field is the condition. Depending on wether the data
     is speculative or not, $dataOperand will be send to $trueResult or 
     $falseResult. The speculative tag is a fictious attribute.

     Example:
     ```mlir
     %trueResult, %falseResult = 
       speculating_branch[%tagFromOperand] %dataOperand : i32, i1
     ```
  }];

  let arguments = (ins AnyType : $tagFromOperand, AnyType : $dataOperand);
  let results = (outs AnyType : $trueResult, AnyType : $falseResult);
  let assemblyFormat = " `[` $tagFromOperand `]` $dataOperand attr-dict `:` qualified(type($tagFromOperand)) `,` qualified(type($dataOperand)) ";
}
