cmake_minimum_required(VERSION 3.20.6...3.22.6 FATAL_ERROR)

project(AI_Agent)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/DCBIA-OrthoLab/AI_Agent")
set(EXTENSION_CONTRIBUTORS "Alexandre Buisson (University of North Carolina)")
set(EXTENSION_DESCRIPTION "AI Agent for 3D Slicer: describe a dental/orthodontic imaging task in natural language and let a locally-run LLM (qwen3:8b via Ollama) pick the right tool, extract its parameters and run it.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/DCBIA-OrthoLab/AI_Agent/main/AI_Agent.png")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(Agent_CLI)
add_subdirectory(Agent_UI)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
