##
#  CMake script for the & tutorial program:
##

# Set the name of the project and target:
SET(TARGET "Test-1")

# Declare all source files the target consists of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
#  FILE(GLOB_RECURSE TARGET_SRC  "source/*.cc")
#  FILE(GLOB_RECURSE TARGET_INC  "include/*.h")
#  SET(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})
# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.
SET(TARGET_SRC
  ${TARGET}.cc
  )

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 3.3.0)

FIND_PACKAGE(deal.II 9.4.0
  HINTS
    ${deal.II_DIR} ${DEAL_II_DIR} "/home/apratim/Softwares/Deal.II/deal.II-v9.4.2/"  $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF ()

#
# Are all dependencies fulfilled?
#
IF(NOT DEAL_II_WITH_MPI OR NOT DEAL_II_WITH_PETSC OR DEAL_II_PETSC_WITH_COMPLEX) # keep in one line
  MESSAGE(FATAL_ERROR "
Error! This tutorial requires a deal.II library that was configured with the following options:
    DEAL_II_WITH_MPI = ON
    DEAL_II_WITH_PETSC = ON
    DEAL_II_PETSC_WITH_COMPLEX = OFF
However, the deal.II library found at ${DEAL_II_PATH} was configured with these options:
    DEAL_II_WITH_MPI = ${DEAL_II_WITH_MPI}
    DEAL_II_WITH_PETSC = ${DEAL_II_WITH_PETSC}
    DEAL_II_PETSC_WITH_COMPLEX = ${DEAL_II_PETSC_WITH_COMPLEX}
This conflicts with the requirements."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()
