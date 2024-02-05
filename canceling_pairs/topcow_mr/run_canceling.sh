#!/bin/bash

./build/CancelPairs -px 10 -py 10 -pz 10 -t 1 -od "<" -ov "<" -p -s foregrounds data/topcow_mr/foregrounds
./build/CancelPairs -px 10 -py 10 -pz 10 -t 1 -od ">" -ov ">" -p -s foregrounds data/topcow_mr/foregrounds
./build/CancelPairs -px 10 -py 10 -pz 10 -t 1 -od "<" -ov ">" -p -s foregrounds data/topcow_mr/foregrounds
./build/CancelPairs -px 10 -py 10 -pz 10 -t 1 -od ">" -ov "<" -p -s foregrounds data/topcow_mr/foregrounds

./build/CancelPairs -px 10 -py 10 -pz 10 -t 1 -od "<" -ov "<" -p -s backgrounds data/topcow_mr/backgrounds
./build/CancelPairs -px 10 -py 10 -pz 10 -t 1 -od ">" -ov ">" -p -s backgrounds data/topcow_mr/backgrounds
./build/CancelPairs -px 10 -py 10 -pz 10 -t 1 -od "<" -ov ">" -p -s backgrounds data/topcow_mr/backgrounds
./build/CancelPairs -px 10 -py 10 -pz 10 -t 1 -od ">" -ov "<" -p -s backgrounds data/topcow_mr/backgrounds

./build/CancelPairs -px 10 -py 10 -pz 10 -t 0 -od "<" -ov "<" -p -s distances_foregrounds data/topcow_mr/distances_foregrounds
./build/CancelPairs -px 10 -py 10 -pz 10 -t 0 -od ">" -ov ">" -p -s distances_foregrounds data/topcow_mr/distances_foregrounds
./build/CancelPairs -px 10 -py 10 -pz 10 -t 0 -od "<" -ov ">" -p -s distances_foregrounds data/topcow_mr/distances_foregrounds
./build/CancelPairs -px 10 -py 10 -pz 10 -t 0 -od ">" -ov "<" -p -s distances_foregrounds data/topcow_mr/distances_foregrounds

./build/CancelPairs -px 10 -py 10 -pz 10 -t 0 -od "<" -ov "<" -p -s distances_backgrounds data/topcow_mr/distances_backgrounds
./build/CancelPairs -px 10 -py 10 -pz 10 -t 0 -od ">" -ov ">" -p -s distances_backgrounds data/topcow_mr/distances_backgrounds
./build/CancelPairs -px 10 -py 10 -pz 10 -t 0 -od "<" -ov ">" -p -s distances_backgrounds data/topcow_mr/distances_backgrounds
./build/CancelPairs -px 10 -py 10 -pz 10 -t 0 -od ">" -ov "<" -p -s distances_backgrounds data/topcow_mr/distances_backgrounds