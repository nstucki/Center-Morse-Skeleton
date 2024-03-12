!/bin/bash

./build/CancelPairs -px 5 -py 5 -pz 5 -t 1 -od ">" -ov ">" -p -s foregrounds data/topcow_mr/foregrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 1 -od ">" -ov "<" -p -s foregrounds data/topcow_mr/foregrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 1 -od "<" -ov ">" -p -s foregrounds data/topcow_mr/foregrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 1 -od "<" -ov "<" -p -s foregrounds data/topcow_mr/foregrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 1 -od "><" -ov ">" -p -s foregrounds data/topcow_mr/foregrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 1 -od "><" -ov "<" -p -s foregrounds data/topcow_mr/foregrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 1 -od "<>" -ov ">" -p -s foregrounds data/topcow_mr/foregrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 1 -od "<>" -ov "<" -p -s foregrounds data/topcow_mr/foregrounds

./build/CancelPairs -px 5 -py 5 -pz 5 -t 0 -od ">" -ov ">" -p -s distances_foregrounds data/topcow_mr/distances_foregrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 0 -od ">" -ov "<" -p -s distances_foregrounds data/topcow_mr/distances_foregrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 0 -od "<" -ov ">" -p -s distances_foregrounds data/topcow_mr/distances_foregrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 0 -od "<" -ov "<" -p -s distances_foregrounds data/topcow_mr/distances_foregrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 0 -od "><" -ov ">" -p -s distances_foregrounds data/topcow_mr/distances_foregrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 0 -od "><" -ov "<" -p -s distances_foregrounds data/topcow_mr/distances_foregrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 0 -od "<>" -ov ">" -p -s distances_foregrounds data/topcow_mr/distances_foregrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 0 -od "<>" -ov "<" -p -s distances_foregrounds data/topcow_mr/distances_foregrounds

./build/CancelPairs -px 5 -py 5 -pz 5 -t 1 -od ">" -ov ">" -p -s backgrounds data/topcow_mr/backgrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 1 -od ">" -ov "<" -p -s backgrounds data/topcow_mr/backgrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 1 -od "<" -ov ">" -p -s backgrounds data/topcow_mr/backgrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 1 -od "<" -ov "<" -p -s backgrounds data/topcow_mr/backgrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 1 -od "><" -ov ">" -p -s backgrounds data/topcow_mr/backgrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 1 -od "><" -ov "<" -p -s backgrounds data/topcow_mr/backgrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 1 -od "<>" -ov ">" -p -s backgrounds data/topcow_mr/backgrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 1 -od "<>" -ov "<" -p -s backgrounds data/topcow_mr/backgrounds

./build/CancelPairs -px 5 -py 5 -pz 5 -t 0 -od ">" -ov ">" -p -s distances_backgrounds data/topcow_mr/distances_backgrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 0 -od ">" -ov "<" -p -s distances_backgrounds data/topcow_mr/distances_backgrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 0 -od "<" -ov ">" -p -s distances_backgrounds data/topcow_mr/distances_backgrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 0 -od "<" -ov "<" -p -s distances_backgrounds data/topcow_mr/distances_backgrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 0 -od "><" -ov ">" -p -s distances_backgrounds data/topcow_mr/distances_backgrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 0 -od "><" -ov "<" -p -s distances_backgrounds data/topcow_mr/distances_backgrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 0 -od "<>" -ov ">" -p -s distances_backgrounds data/topcow_mr/distances_backgrounds
./build/CancelPairs -px 5 -py 5 -pz 5 -t 0 -od "<>" -ov "<" -p -s distances_backgrounds data/topcow_mr/distances_backgrounds