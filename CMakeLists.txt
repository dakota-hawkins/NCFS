find_package(xtensor REQUIRED)
target_include_directories(NCFS PUBLIC ${xtensor_INCLUDE_DIRS})
target_link_libraries(NCFS PUBLIC xtensor)