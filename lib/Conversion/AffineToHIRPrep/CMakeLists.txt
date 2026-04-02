add_circt_conversion_library(CIRCTAffineToHIRPrep
  AffineToHIRPrep.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRArithDialect
  CIRCTHIR
)
