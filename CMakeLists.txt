cmake_minimum_required(VERSION 3.14)
project(New_folder)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(New_folder
        Barrier.cpp
        Barrier.h
        Makefile
        Client.h
        Framework.cpp
        Framework.h main.cpp)
