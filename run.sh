#!/bin/bash
cd build
cmake ..
make
cd ..
~/robot/bitbot-frontend/bitbot-frontend --no-sandbox &
APP1_PID=$!

./build/bin/main_app

kill $APP1_PID



