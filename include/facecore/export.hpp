#pragma once

#if defined(_WIN32) && defined(FACECORE_SHARED)
#  if defined(FACECORE_BUILDING_LIBRARY)
#    define FACECORE_API __declspec(dllexport)
#  else
#    define FACECORE_API __declspec(dllimport)
#  endif
#else
#  define FACECORE_API
#endif

