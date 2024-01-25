#!/bin/bash

./../../build/CancelPairs -px 10 -py 10 -pz 10 -t 1 -od "<" -ov "<" -p -s foregrounds ../../data/synthetic_blob_data_holes/foregrounds
./../../build/CancelPairs -px 10 -py 10 -pz 10 -t 1 -od ">" -ov ">" -p -s foregrounds ../../data/synthetic_blob_data_holes/foregrounds
./../../build/CancelPairs -px 10 -py 10 -pz 10 -t 1 -od "<" -ov ">" -p -s foregrounds ../../data/synthetic_blob_data_holes/foregrounds
./../../build/CancelPairs -px 10 -py 10 -pz 10 -t 1 -od ">" -ov "<" -p -s foregrounds ../../data/synthetic_blob_data_holes/foregrounds

./../../build/CancelPairs -px 10 -py 10 -pz 10 -t 1 -od "<" -ov "<" -p -s backgrounds ../../data/synthetic_blob_data_holes/backgrounds
./../../build/CancelPairs -px 10 -py 10 -pz 10 -t 1 -od ">" -ov ">" -p -s backgrounds ../../data/synthetic_blob_data_holes/backgrounds
./../../build/CancelPairs -px 10 -py 10 -pz 10 -t 1 -od "<" -ov ">" -p -s backgrounds ../../data/synthetic_blob_data_holes/backgrounds
./../../build/CancelPairs -px 10 -py 10 -pz 10 -t 1 -od ">" -ov "<" -p -s backgrounds ../../data/synthetic_blob_data_holes/backgrounds

./../../build/CancelPairs -px 10 -py 10 -pz 10 -t 0 -od "<" -ov "<" -p -s distances_foregrounds ../../data/synthetic_blob_data_holes/distances_foregrounds
./../../build/CancelPairs -px 10 -py 10 -pz 10 -t 0 -od ">" -ov ">" -p -s distances_foregrounds ../../data/synthetic_blob_data_holes/distances_foregrounds
./../../build/CancelPairs -px 10 -py 10 -pz 10 -t 0 -od "<" -ov ">" -p -s distances_foregrounds ../../data/synthetic_blob_data_holes/distances_foregrounds
./../../build/CancelPairs -px 10 -py 10 -pz 10 -t 0 -od ">" -ov "<" -p -s distances_foregrounds ../../data/synthetic_blob_data_holes/distances_foregrounds

./../../build/CancelPairs -px 10 -py 10 -pz 10 -t 0 -od "<" -ov "<" -p -s distances_backgrounds ../../data/synthetic_blob_data_holes/distances_backgrounds
./../../build/CancelPairs -px 10 -py 10 -pz 10 -t 0 -od ">" -ov ">" -p -s distances_backgrounds ../../data/synthetic_blob_data_holes/distances_backgrounds
./../../build/CancelPairs -px 10 -py 10 -pz 10 -t 0 -od "<" -ov ">" -p -s distances_backgrounds ../../data/synthetic_blob_data_holes/distances_backgrounds
./../../build/CancelPairs -px 10 -py 10 -pz 10 -t 0 -od ">" -ov "<" -p -s distances_backgrounds ../../data/synthetic_blob_data_holes/distances_backgrounds