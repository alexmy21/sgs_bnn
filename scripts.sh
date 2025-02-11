!#/bin/bash
cat src/my_lib.lua | redis-cli -x FUNCTION LOAD REPLACE